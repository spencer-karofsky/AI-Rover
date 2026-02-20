#!/usr/bin/env python3
"""
Depth Fusion for AI Rover.
Combines SCDepthV3 on Hailo (dense, relative) with stereo SGBM (sparse, metric).

Key optimization: SGBM only runs every anchor_interval frames (default 10).
All other frames only grab left camera + run Hailo — no SGBM cost.
"""
import os
import numpy as np
import cv2
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import threading

from hailo_platform import VDevice, HailoSchedulingAlgorithm
from stereo import StereoCamera

os.environ['DISPLAY'] = ':0'


@dataclass
class FusedFrame:
    rgb:            np.ndarray   # Left camera RGB (640x480)
    depth_relative: np.ndarray   # Raw SCDepthV3 uint16 (256x320)
    depth_metric:   np.ndarray   # Fused metric depth mm (640x480)
    depth_stereo:   Optional[np.ndarray]  # Sparse stereo depth mm — None on non-anchor frames
    colorized:      np.ndarray   # BGR colormap of fused depth (640x480)
    hailo_ms:       float
    stereo_ms:      float        # 0.0 on non-anchor frames
    scale:          float
    offset:         float
    anchor_age:     int


class DepthFusion:
    DEFAULT_HEF = '/home/spencer/ai_rover/scdepthv3.hef'

    def __init__(
        self,
        hef_path: str = DEFAULT_HEF,
        calib_path: str = 'stereo_calib.npz',
        resolution: tuple[int, int] = (640, 480),
        cam_left: int = 1,
        cam_right: int = 0,
        anchor_interval: int = 10,   # run SGBM every N frames
        min_anchor_points: int = 100,
        min_stereo_mm: float = 100,
        max_stereo_mm: float = 5000,
    ):
        self._resolution       = resolution
        self._anchor_interval  = anchor_interval
        self._min_anchor_points = min_anchor_points
        self._min_stereo_mm    = min_stereo_mm
        self._max_stereo_mm    = max_stereo_mm
        self._scale            = 1.0
        self._offset           = 0.0
        self._anchor_age       = 0
        self._frame_count      = 0
        self._running          = False
        self._last_stereo_mm   = None  # cached from last SGBM run

        # Background SGBM thread
        self._sgbm_thread  = None
        self._sgbm_running = False
        self._sgbm_result  = None      # latest depth_mm from background SGBM
        self._sgbm_lock    = threading.Lock()
        self._sgbm_trigger = threading.Event()  # signals thread to run SGBM

        # --- Hailo ---
        print(f"Loading Hailo model: {Path(hef_path).name}")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        self._vdevice      = VDevice(params)
        self._infer_model  = self._vdevice.create_infer_model(hef_path)
        self._infer_model.set_batch_size(1)
        self._configured   = self._infer_model.configure()
        self._output_name  = self._infer_model.output().name
        self._output_shape = self._infer_model.output().shape
        self._output_buffer = np.empty(self._output_shape, dtype=np.uint16)
        print(f"  Hailo input:  {self._infer_model.input().shape}")
        print(f"  Hailo output: {self._output_shape}")

        # --- Stereo ---
        print("Loading stereo system...")
        self._stereo = StereoCamera(
            calib_path=calib_path,
            resolution=resolution,
            cam_left=cam_left,
            cam_right=cam_right,
        )
        print("Fusion ready.")

    def start(self):
        if not self._running:
            self._stereo.start()
            self._running    = True
            self._sgbm_running = True
            self._sgbm_thread = threading.Thread(target=self._sgbm_loop, daemon=True)
            self._sgbm_thread.start()

    def stop(self):
        if self._running:
            # Stop SGBM background thread first
            self._sgbm_running = False
            self._sgbm_trigger.set()
            if self._sgbm_thread:
                self._sgbm_thread.join(timeout=2.0)

            self._running = False
            self._stereo.stop()

            # Explicitly release Hailo resources in order
            time.sleep(0.2)
            try:
                del self._configured
                del self._infer_model
                del self._vdevice
            except Exception:
                pass

    def _sgbm_loop(self):
        """Background thread: runs SGBM whenever triggered, never blocks capture()."""
        while self._sgbm_running:
            self._sgbm_trigger.wait()   # sleep until capture() triggers us
            self._sgbm_trigger.clear()
            if not self._sgbm_running:
                break
            try:
                stereo_frame = self._stereo.capture()
                with self._sgbm_lock:
                    self._sgbm_result = stereo_frame.depth_mm
            except Exception as e:
                print(f"[SGBM] Error: {e}")

    def capture(self) -> FusedFrame:
        self._frame_count += 1
        self._anchor_age  += 1

        # Always: left camera only + rectify — never blocks on SGBM
        frame_rgb, _ = self._stereo.capture_raw()
        frame_rgb = cv2.remap(
            frame_rgb,
            self._stereo._map_lx, self._stereo._map_ly,
            cv2.INTER_LINEAR
        )

        # Trigger background SGBM every anchor_interval frames (non-blocking)
        if self._anchor_age >= self._anchor_interval:
            self._sgbm_trigger.set()
            self._anchor_age = 0

        # Hailo inference
        t1 = time.perf_counter()
        hailo_input = cv2.resize(frame_rgb, (320, 256), interpolation=cv2.INTER_LINEAR)
        bindings    = self._configured.create_bindings(
            output_buffers={self._output_name: self._output_buffer}
        )
        bindings.input().set_buffer(hailo_input)
        self._configured.run([bindings], timeout=1000)
        depth_relative = self._output_buffer.squeeze().copy()
        hailo_ms       = (time.perf_counter() - t1) * 1000

        # Check for fresh SGBM result from background thread
        with self._sgbm_lock:
            stereo_depth = self._sgbm_result
            self._sgbm_result = None  # consume it

        if stereo_depth is not None:
            self._update_anchor(depth_relative, stereo_depth)

        # Fuse
        d = depth_relative.astype(np.float32)
        if self._scale != 1.0 or self._offset != 0.0:
            depth_metric = np.clip(d * self._scale + self._offset, 0, 10000)
            d_log  = np.log1p(np.clip(depth_metric, 0, 5000))
            d_norm = (d_log / np.log1p(5000) * 255).astype(np.uint8)
        else:
            depth_metric = d
            d_log  = np.log1p(d - d.min())
            d_norm = (d_log / (d_log.max() + 1e-6) * 255).astype(np.uint8)

        depth_metric    = cv2.resize(depth_metric, (640, 480))
        colorized_input = cv2.resize(d_norm, (640, 480))
        colorized_input = cv2.equalizeHist(colorized_input)
        colorized       = cv2.applyColorMap(colorized_input, cv2.COLORMAP_TURBO)

        return FusedFrame(
            rgb=frame_rgb,
            depth_relative=depth_relative,
            depth_metric=depth_metric,
            depth_stereo=stereo_depth,
            colorized=colorized,
            hailo_ms=hailo_ms,
            stereo_ms=0.0,  # always 0 — SGBM is background
            scale=self._scale,
            offset=self._offset,
            anchor_age=self._anchor_age,
        )

    def _update_anchor(self, depth_relative: np.ndarray, stereo_mm: np.ndarray):
        h_r, w_r    = depth_relative.shape
        stereo_small = cv2.resize(stereo_mm, (w_r, h_r), interpolation=cv2.INTER_NEAREST)

        valid = (stereo_small >= self._min_stereo_mm) & (stereo_small <= self._max_stereo_mm)
        if np.sum(valid) < self._min_anchor_points:
            return

        rel    = depth_relative[valid].astype(np.float64)
        metric = stereo_small[valid].astype(np.float64)

        rel_mean    = np.mean(rel)
        metric_mean = np.mean(metric)
        cov = np.mean((rel - rel_mean) * (metric - metric_mean))
        var = np.var(rel)
        if var < 1e-6:
            return

        new_scale  = cov / var
        new_offset = metric_mean - new_scale * rel_mean
        if new_scale <= 0 or new_scale > 1000:
            return

        alpha        = 0.3
        self._scale  = alpha * new_scale  + (1 - alpha) * self._scale
        self._offset = alpha * new_offset + (1 - alpha) * self._offset
        print(f"  Anchor: scale={self._scale:.3f} offset={self._offset:.1f} "
              f"stereo_ms={0:.0f} points={np.sum(valid)}")

    def get_obstacle_distances_mm(
        self,
        depth_metric: np.ndarray,
        regions: int = 3,
        min_valid_mm: float = 100,
        max_valid_mm: float = 5000,
    ) -> list[Optional[float]]:
        h, w = depth_metric.shape
        rw   = w // regions
        results = []
        for i in range(regions):
            x0     = i * rw
            x1     = (i + 1) * rw if i < regions - 1 else w
            region = depth_metric[:, x0:x1]
            valid  = region[(region >= min_valid_mm) & (region <= max_valid_mm)]
            results.append(float(np.median(valid)) if valid.size > 0 else None)
        return results

    def check_path_clear(
        self,
        depth_metric: np.ndarray,
        obstacle_threshold_mm: float = 600,
    ) -> tuple[bool, str]:
        l, c, r = self.get_obstacle_distances_mm(depth_metric)
        def safe(d): return d is None or d > obstacle_threshold_mm
        if safe(c): return True,  'forward'
        if safe(l): return False, 'left'
        if safe(r): return False, 'right'
        return False, 'reverse'

    def create_debug_view(self, frame: FusedFrame) -> np.ndarray:
        h, w    = frame.rgb.shape[:2]
        rgb_bgr = frame.rgb
        dists   = self.get_obstacle_distances_mm(frame.depth_metric)
        labels  = ['L', 'C', 'R']
        rw      = w // 3

        for i, (lbl, d) in enumerate(zip(labels, dists)):
            x = 10 + i * rw
            if d is not None:
                color = (0, 255, 0) if d > 600 else (0, 165, 255) if d > 300 else (0, 0, 255)
                cv2.putText(rgb_bgr, f'{lbl}:{d/1000:.1f}m',
                            (x, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            else:
                cv2.putText(rgb_bgr, f'{lbl}:--',
                            (x, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

        depth_vis = frame.colorized.copy()
        stereo_str = f'S:{frame.stereo_ms:.0f}ms' if frame.stereo_ms > 0 else 'S:skip'
        cv2.putText(depth_vis,
                    f'H:{frame.hailo_ms:.0f}ms {stereo_str} scale:{frame.scale:.2f}',
                    (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return np.hstack([rgb_bgr, depth_vis])