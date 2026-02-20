#!/usr/bin/env python3
"""
StereoNet Depth Fusion for AI Rover (Hailo-10H Optimized).
Replaces SCDepthV3 + SGBM with a full NPU-accelerated stereo pipeline.
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

@dataclass
class FusedFrame:
    rgb:            np.ndarray   # Rectified Left RGB (640x480)
    depth_metric:   np.ndarray   # Metric depth mm (640x480)
    colorized:      np.ndarray   # Visualization
    hailo_ms:       float
    focal_length:   float
    baseline_mm:    float

class DepthFusion:
    DEFAULT_HEF = '/home/spencer/ai_rover/stereonet.hef'

    def __init__(
        self,
        hef_path: str = DEFAULT_HEF,
        calib_path: str = 'stereo_calib.npz',
        resolution: tuple[int, int] = (640, 480),
    ):
        self._resolution = resolution
        
        # --- Calibration Data ---
        # We need Q and focal length to convert disparity to mm
        if os.path.exists(calib_path):
            c = np.load(calib_path)
            self._focal_length = c['K1'][0, 0]  # fx
            self._baseline_mm = abs(float(c['T'][0].item()))
            print(f"Loaded Calibration: f={self._focal_length:.1f}, B={self._baseline_mm:.1f}mm")
        else:
            print("Warning: calib_path not found, using defaults.")
            self._focal_length = 500.0
            self._baseline_mm = 60.0

        # --- Hailo-10H Setup ---
        print(f"Loading StereoNet HEF: {Path(hef_path).name}")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = VDevice(params)
        self._infer_model = self._vdevice.create_infer_model(hef_path)
        self._infer_model.set_batch_size(1)
        
        self._configured = self._infer_model.configure()
        
        # Mapping names from your 'parse-hef' output
        self._input_left_name  = "stereonet/input_layer1"
        self._input_right_name = "stereonet/input_layer2"
        self._output_name      = "stereonet/conv53"
        
        # Internal buffers for the 1232x368 requirement
        self._hailo_width  = 1232
        self._hailo_height = 368
        
        print("StereoNet Fusion Ready (Hailo-10H)")

    def capture(self, left_rect: np.ndarray, right_rect: np.ndarray) -> FusedFrame:
        t1 = time.perf_counter()

        # 1. Pre-process (1232x368 for StereoNet)
        in_l = cv2.resize(left_rect, (self._hailo_width, self._hailo_height))
        in_r = cv2.resize(right_rect, (self._hailo_width, self._hailo_height))

        # 2. Hailo Inference
        bindings = self._configured.create_bindings()
        
        # Set Input Buffers
        bindings.input(self._input_left_name).set_buffer(in_l)
        bindings.input(self._input_right_name).set_buffer(in_r)
        
        # Pre-allocate output buffer (Prevents the 'view' error)
        output_buffer = np.empty((self._hailo_height, self._hailo_width, 1), dtype=np.uint8)
        bindings.output(self._output_name).set_buffer(output_buffer)
        
        # Execute (Wrapped in a LIST to fix 'not iterable' error)
        # Timeout set to 1000ms
        self._configured.run([bindings], 1000)
        
        # After run, the output_buffer is populated via DMA
        disparity = output_buffer.squeeze().astype(np.float32) / 16.0
        hailo_ms = (time.perf_counter() - t1) * 1000

        # 3. Post-process (Scale and Triangulate)
        safe_disp = np.maximum(disparity, 0.1)
        
        # IMPORTANT: Scaling focal length for 1232 width model
        # 1232 / 640 = 1.925x scaling
        scale_factor = self._hailo_width / self._resolution[0] 
        f_scaled = self._focal_length * scale_factor
        
        depth_metric = (f_scaled * self._baseline_mm) / safe_disp
        
        # Resize back to 640x480 for the rover's SLAM/Nav stack
        depth_metric_resized = cv2.resize(depth_metric, self._resolution, interpolation=cv2.INTER_NEAREST)

        # 4. Colorization
        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        colorized = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)

        return FusedFrame(
            rgb=left_rect,
            depth_metric=depth_metric_resized,
            colorized=colorized,
            hailo_ms=hailo_ms,
            focal_length=self._focal_length,
            baseline_mm=self._baseline_mm
        )
    
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

    # def create_debug_view(self, frame: FusedFrame) -> np.ndarray:
    #     # 1. Ensure we have a BGR version of the left camera
    #     rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    #     h, w = rgb_bgr.shape[:2]

    #     # 2. Add Obstacle Labels
    #     dists = self.get_obstacle_distances_mm(frame.depth_metric)
    #     labels = ['L', 'C', 'R']
    #     for i, (lbl, d) in enumerate(zip(labels, dists)):
    #         x_pos = 20 + i * (w // 3)
    #         if d:
    #             cv2.putText(rgb_bgr, f"{lbl}: {d/1000:.2f}m", (x_pos, h-20), 
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    #     # 3. Process Depth Map for display
    #     # Resize it to match the camera height exactly
    #     depth_vis = cv2.resize(frame.colorized, (w, h))

    #     # 4. DRAW ALIGNMENT LINES (Crucial for "Dead Accuracy")
    #     # These lines help you see if your rectification is actually level
    #     for y in range(50, h, 50):
    #         cv2.line(rgb_bgr, (0, y), (w, y), (0, 255, 255), 1)
    #         cv2.line(depth_vis, (0, y), (w, y), (0, 255, 255), 1)

    #     # 5. Combine Side-by-Side
    #     combined = np.hstack([rgb_bgr, depth_vis])

    #     # 6. Performance Overlay
    #     cv2.putText(combined, f"Hailo Inference: {frame.hailo_ms:.1f}ms", (20, 40), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    #     return combined
    def create_debug_view(self, frame_data, right_rect, show_lines: bool = True):
        """
        Diagnostic view: [Left Rectified] [Right Rectified] [Depth Map]
        show_lines: draw yellow epipolar alignment lines across all panels
        """
        h, w = frame_data.rgb.shape[:2]

        l_bgr = cv2.cvtColor(frame_data.rgb, cv2.COLOR_RGB2BGR)
        r_bgr = cv2.cvtColor(right_rect, cv2.COLOR_RGB2BGR)
        depth_vis = cv2.resize(frame_data.colorized, (w, h))

        if show_lines:
            for y in range(40, h, 60):
                cv2.line(l_bgr,     (0, y), (w, y), (0, 255, 255), 1)
                cv2.line(r_bgr,     (0, y), (w, y), (0, 255, 255), 1)
                cv2.line(depth_vis, (0, y), (w, y), (0, 255, 255), 1)

        cv2.putText(l_bgr,     "LEFT RECT",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(r_bgr,     "RIGHT RECT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(depth_vis, "DEPTH",      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return np.hstack([l_bgr, r_bgr, depth_vis])