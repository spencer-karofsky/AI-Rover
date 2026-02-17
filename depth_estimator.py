#!/usr/bin/env python3
"""
Vision module for AI Rover.
Handles camera capture + Hailo-accelerated depth estimation.
"""
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from picamera2 import Picamera2
from hailo_platform import VDevice, HailoSchedulingAlgorithm

# Optional metric depth (lazy loaded)
_metric_model = None
_metric_processor = None


@dataclass
class DepthFrame:
    """Container for depth estimation results."""
    rgb: np.ndarray          # Original RGB frame
    raw: np.ndarray          # Raw uint16 depth map from model
    normalized: np.ndarray   # 0-1 float, 0=close, 1=far
    colorized: np.ndarray    # BGR colormap visualization
    metric: Optional[np.ndarray]  # Metric depth in mm (if calibrated)
    inference_ms: float      # Inference time in milliseconds


class DepthEstimator:
    """
    Camera + Hailo-accelerated monocular depth estimation.
    
    Usage:
        depth = DepthEstimator()
        depth.start()
        
        frame = depth.capture()  # Returns DepthFrame
        left, center, right = depth.get_obstacle_distances(frame.normalized)
        
        depth.stop()
    """
    
    DEFAULT_HEF = '/home/spencer/ai_rover/scdepthv3.hef'
    
    def __init__(
        self,
        hef_path: str = DEFAULT_HEF,
        resolution: tuple[int, int] = (320, 256),
        colormap: int = cv2.COLORMAP_MAGMA,
        flip_vertical: bool = True
    ):
        """
        Initialize camera and depth estimator.
        
        Args:
            hef_path: Path to compiled Hailo model (.hef)
            resolution: Camera resolution (width, height), must match model input
            colormap: OpenCV colormap for visualization
            flip_vertical: Flip camera image vertically (True if camera mounted upside down)
        """
        self._hef_path = Path(hef_path)
        if not self._hef_path.exists():
            raise FileNotFoundError(f"HEF not found: {hef_path}")
        
        self._resolution = resolution
        self._colormap = colormap
        self._flip = flip_vertical
        
        # Camera setup
        print("Initializing camera...")
        self._camera = Picamera2()
        config = self._camera.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        self._camera.configure(config)
        
        # Load Hailo model
        print(f"Loading depth model: {self._hef_path.name}")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        
        self._vdevice = VDevice(params)
        self._infer_model = self._vdevice.create_infer_model(str(self._hef_path))
        self._infer_model.set_batch_size(1)
        self._configured = self._infer_model.configure()
        
        # Cache model info
        self._input_shape = self._infer_model.input().shape
        self._output_name = self._infer_model.output().name
        self._output_shape = self._infer_model.output().shape
        
        # Pre-allocate output buffer
        self._output_buffer = np.empty(self._output_shape, dtype=np.uint16)
        
        # State
        self._running = False
        self._latest_frame: Optional[DepthFrame] = None
        
        # Metric calibration state
        self._metric_scale = 1.0      # Multiply raw by this to get mm
        self._metric_offset = 0.0     # Add this after scaling
        self._last_calibration = 0.0  # Timestamp of last metric update
        self._calibration_interval = 3.0  # Seconds between metric updates
        self._use_metric = False      # Whether metric model is loaded
        
        print(f"  Resolution: {resolution[0]}x{resolution[1]}")
        print(f"  Model input:  {self._input_shape}")
        print(f"  Model output: {self._output_shape}")
        print("Vision ready.")
    
    def start(self):
        """Start the camera."""
        if not self._running:
            self._camera.start()
            self._running = True
    
    def stop(self):
        """Stop the camera."""
        if self._running:
            self._camera.stop()
            self._running = False
    
    def enable_metric_depth(self, calibration_interval: float = 3.0):
        """
        Enable metric depth using Depth Anything V2 for calibration.
        This loads the model on first call (slow, ~2-3 seconds).
        
        Args:
            calibration_interval: Seconds between metric recalibrations
        """
        global _metric_model, _metric_processor
        
        if _metric_model is None:
            print("Loading Depth Anything V2 for metric calibration...")
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                import torch
                
                _metric_processor = AutoImageProcessor.from_pretrained(
                    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                    use_fast=True
                )
                _metric_model = AutoModelForDepthEstimation.from_pretrained(
                    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
                )
                _metric_model.eval()
                print("  Metric depth model loaded")
            except Exception as e:
                print(f"  Failed to load metric model: {e}")
                return
        
        self._use_metric = True
        self._calibration_interval = calibration_interval
        self._last_calibration = 0  # Force immediate calibration
        print(f"  Metric depth enabled (recalibrate every {calibration_interval}s)")
    
    def _update_metric_calibration(self, frame_rgb: np.ndarray, raw_depth: np.ndarray):
        """Update metric calibration using Depth Anything V2."""
        global _metric_model, _metric_processor
        
        if not self._use_metric or _metric_model is None:
            return
        
        now = time.perf_counter()
        if now - self._last_calibration < self._calibration_interval:
            return
        
        try:
            from PIL import Image
            import torch
            
            # Run metric model
            pil_image = Image.fromarray(frame_rgb)
            inputs = _metric_processor(images=pil_image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = _metric_model(**inputs)
            
            metric_depth = outputs.predicted_depth.squeeze().cpu().numpy()
            # Resize to match our depth map
            metric_depth = cv2.resize(metric_depth, (raw_depth.shape[1], raw_depth.shape[0]))
            
            # Metric model outputs meters, convert to mm
            metric_mm = metric_depth * 1000
            
            # Compute scale/offset using linear regression on center region
            h, w = raw_depth.shape
            margin_h, margin_w = h // 4, w // 4
            raw_center = raw_depth[margin_h:-margin_h, margin_w:-margin_w].flatten().astype(np.float64)
            metric_center = metric_mm[margin_h:-margin_h, margin_w:-margin_w].flatten()
            
            # Simple linear fit: metric = scale * raw + offset
            # Use least squares: scale = cov(raw, metric) / var(raw)
            raw_mean = np.mean(raw_center)
            metric_mean = np.mean(metric_center)
            
            cov = np.mean((raw_center - raw_mean) * (metric_center - metric_mean))
            var = np.var(raw_center)
            
            if var > 1e-6:
                self._metric_scale = cov / var
                self._metric_offset = metric_mean - self._metric_scale * raw_mean
            
            self._last_calibration = now
            
        except Exception as e:
            print(f"Metric calibration error: {e}")
    
    @property
    def resolution(self) -> tuple[int, int]:
        """Camera resolution as (width, height)."""
        return self._resolution
    
    @property
    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._running
    
    def capture_rgb(self) -> np.ndarray:
        """Capture RGB frame only (no depth inference)."""
        frame = self._camera.capture_array()
        if self._flip:
            frame = cv2.flip(frame, -1)  # -1 = both axes (vertical + horizontal)
        return frame
    
    def capture(self) -> DepthFrame:
        """
        Capture frame and run depth estimation.
        
        Returns:
            DepthFrame with RGB, raw depth, normalized depth, and colorized depth
        """
        import time
        
        # Capture RGB
        frame_rgb = self.capture_rgb()
        
        # Run inference
        start = time.perf_counter()
        
        bindings = self._configured.create_bindings(
            output_buffers={self._output_name: self._output_buffer}
        )
        bindings.input().set_buffer(frame_rgb)
        self._configured.run([bindings], timeout=1000)
        
        raw = self._output_buffer.squeeze().copy()
        inference_ms = (time.perf_counter() - start) * 1000
        
        # Normalize to 0-1 (lower raw values = closer)
        d = raw.astype(np.float32)
        d_min, d_max = d.min(), d.max()
        normalized = (d - d_min) / (d_max - d_min + 1e-6)
        
        # Update metric calibration periodically
        self._update_metric_calibration(frame_rgb, raw)
        
        # Compute metric depth if calibrated
        metric = None
        if self._use_metric:
            metric = (raw.astype(np.float32) * self._metric_scale + self._metric_offset).astype(np.float32)
            metric = np.clip(metric, 0, 10000)  # Clamp to 0-10m
        
        # Colorize
        colorized = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8),
            self._colormap
        )
        
        self._latest_frame = DepthFrame(
            rgb=frame_rgb,
            raw=raw,
            normalized=normalized,
            colorized=colorized,
            metric=metric,
            inference_ms=inference_ms
        )
        
        return self._latest_frame
    
    @property
    def latest(self) -> Optional[DepthFrame]:
        """Get the most recent captured frame."""
        return self._latest_frame
    
    # === Obstacle Detection Helpers ===
    
    def get_obstacle_distances(
        self,
        depth_normalized: np.ndarray,
        regions: int = 3
    ) -> list[float]:
        """
        Get average distances for horizontal regions of the depth map.
        
        Args:
            depth_normalized: Normalized depth map (0=close, 1=far)
            regions: Number of horizontal regions (default 3 = left/center/right)
            
        Returns:
            List of mean distances per region (0=close, 1=far)
        """
        h, w = depth_normalized.shape
        region_width = w // regions
        
        distances = []
        for i in range(regions):
            x_start = i * region_width
            x_end = (i + 1) * region_width if i < regions - 1 else w
            region = depth_normalized[:, x_start:x_end]
            distances.append(float(np.mean(region)))
        
        return distances
    
    def get_center_distance(
        self,
        depth_normalized: np.ndarray,
        width_fraction: float = 0.3,
        height_fraction: float = 0.5
    ) -> float:
        """
        Get distance in center region of frame.
        
        Args:
            depth_normalized: Normalized depth map
            width_fraction: Width of center region as fraction of frame
            height_fraction: Height of center region as fraction of frame
            
        Returns:
            Mean distance in center region (0=close, 1=far)
        """
        h, w = depth_normalized.shape
        
        cx, cy = w // 2, h // 2
        hw = int(w * width_fraction / 2)
        hh = int(h * height_fraction / 2)
        
        center = depth_normalized[cy-hh:cy+hh, cx-hw:cx+hw]
        return float(np.mean(center))
    
    def get_center_distance_mm(self, depth_frame: Optional[DepthFrame] = None) -> Optional[float]:
        """
        Get metric distance to center of frame in millimeters.
        
        Returns:
            Distance in mm, or None if metric depth not available
        """
        frame = depth_frame or self._latest_frame
        if frame is None or frame.metric is None:
            return None
        
        h, w = frame.metric.shape
        cx, cy = w // 2, h // 2
        margin = 20
        
        center = frame.metric[cy-margin:cy+margin, cx-margin:cx+margin]
        return float(np.mean(center))
    
    def find_closest_point(
        self,
        depth_normalized: np.ndarray,
        percentile: float = 5.0
    ) -> tuple[int, int, float]:
        """
        Find the closest point in the depth map.
        
        Args:
            depth_normalized: Normalized depth map
            percentile: Use this percentile to filter noise
            
        Returns:
            (x, y, distance) of closest region centroid
        """
        threshold = np.percentile(depth_normalized, percentile)
        close_mask = depth_normalized <= threshold
        
        if not np.any(close_mask):
            h, w = depth_normalized.shape
            return (w // 2, h // 2, 1.0)
        
        ys, xs = np.where(close_mask)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        dist = float(depth_normalized[cy, cx])
        
        return (cx, cy, dist)
    
    def check_path_clear(
        self,
        depth_normalized: np.ndarray,
        threshold: float = 0.2,
        center_weight: float = 0.6
    ) -> tuple[bool, str]:
        """
        Check if path ahead is clear and suggest action.
        
        Args:
            depth_normalized: Normalized depth map
            threshold: Distance below which obstacle is "close"
            center_weight: How much to weight center vs sides
            
        Returns:
            (is_clear, suggested_action) where action is 'forward', 'left', 'right', or 'reverse'
        """
        left, center, right = self.get_obstacle_distances(depth_normalized)
        
        if center > threshold:
            return (True, 'forward')
        
        # Obstacle ahead - decide turn direction
        if left > right and left > threshold:
            return (False, 'left')
        elif right > threshold:
            return (False, 'right')
        else:
            return (False, 'reverse')
    
    # === Visualization ===
    
    @staticmethod
    def create_colorbar(height: int = 256, width: int = 40) -> np.ndarray:
        """Create a vertical colorbar for depth visualization."""
        colorbar = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            val = int(255 * i / height)
            colorbar[i, :width//2, :] = cv2.applyColorMap(
                np.array([[val]], dtype=np.uint8),
                cv2.COLORMAP_MAGMA
            )[0, 0]
        
        cv2.putText(colorbar, 'near', (width//2 + 2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(colorbar, 'far', (width//2 + 2, height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return colorbar
    
    def create_debug_view(self, frame: DepthFrame) -> np.ndarray:
        """
        Create a side-by-side debug visualization.
        
        Returns:
            Combined RGB + Depth + Colorbar image
        """
        h = frame.rgb.shape[0]
        colorbar = self.create_colorbar(h)
        
        # Annotate depth with timing
        depth_vis = frame.colorized.copy()
        cv2.putText(depth_vis, f'{frame.inference_ms:.1f}ms',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add obstacle distances
        distances = self.get_obstacle_distances(frame.normalized)
        labels = ['L', 'C', 'R']
        for i, (label, dist) in enumerate(zip(labels, distances)):
            x = 10 + i * 100
            color = (0, 255, 0) if dist > 0.3 else (0, 165, 255) if dist > 0.15 else (0, 0, 255)
            cv2.putText(depth_vis, f'{label}:{dist:.2f}',
                        (x, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return np.hstack([frame.rgb, depth_vis, colorbar])


# === Demo ===

def demo():
    """Live depth estimation demo."""
    import os
    os.environ['DISPLAY'] = ':0'
    
    print("Starting depth demo...")
    print("Press 'q' to quit\n")
    
    depth = DepthEstimator()
    depth.start()
    
    cv2.namedWindow("Depth", cv2.WND_PROP_FULLSCREEN)
    
    try:
        while True:
            frame = depth.capture()
            
            # Check path
            clear, action = depth.check_path_clear(frame.normalized)
            status = f"CLEAR -> {action.upper()}" if clear else f"BLOCKED -> {action.upper()}"
            
            # Create visualization
            vis = depth.create_debug_view(frame)
            cv2.putText(vis, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Depth", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        depth.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    demo()