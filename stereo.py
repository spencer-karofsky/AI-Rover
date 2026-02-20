#!/usr/bin/env python3
"""
Stereo vision module for AI Rover.
Handles dual Pi Camera V3 capture, rectification, and disparity computation.
"""
import numpy as np
import cv2
import time
import threading
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from picamera2 import Picamera2


@dataclass
class StereoFrame:
    left: np.ndarray        # Rectified left RGB
    right: np.ndarray       # Rectified right RGB
    disparity: np.ndarray   # Smoothed disparity (float32)
    depth_mm: np.ndarray    # Metric depth in mm, 0 = invalid
    colorized: np.ndarray   # BGR colormap visualization
    capture_ms: float
    disparity_ms: float


class StereoCamera:
    """
    Dual Pi Camera V3 stereo vision.

    Usage:
        stereo = StereoCamera()
        stereo.start()
        frame = stereo.capture()
        stereo.stop()
    """

    def __init__(
        self,
        calib_path: str = 'stereo_calib.npz',
        resolution: tuple[int, int] = (640, 480),
        cam_left: int = 1,
        cam_right: int = 0,
        flip: bool = True,
        use_sgbm: bool = True,
        alpha: float = 0.3,
    ):
        self._res = resolution
        self._flip = flip
        self._running = False
        self._alpha = alpha
        self._smooth_disp = None

        # --- Cameras ---
        print("Initializing stereo cameras...")
        self._cam_l = Picamera2(cam_left)
        self._cam_r = Picamera2(cam_right)
        for cam in (self._cam_l, self._cam_r):
            cfg = cam.create_preview_configuration(
                main={"size": resolution, "format": "RGB888"}
            )
            cam.configure(cfg)

        # --- Calibration ---
        calib_file = Path(calib_path)
        if not calib_file.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {calib_path}\n"
                "Run: python stereo.py calibrate"
            )

        print(f"Loading calibration: {calib_path}")
        c = np.load(calib_file)
        w, h = resolution

        self._map_lx, self._map_ly = cv2.initUndistortRectifyMap(
            c['K1'], c['D1'], c['R1'], c['P1'], (w, h), cv2.CV_32FC1
        )
        self._map_rx, self._map_ry = cv2.initUndistortRectifyMap(
            c['K2'], c['D2'], c['R2'], c['P2'], (w, h), cv2.CV_32FC1
        )
        self._Q = c['Q']
        self._baseline_mm = abs(float(c['T'][0].item()))

        # --- Matcher ---
        self._use_sgbm = use_sgbm
        self._matcher = self._build_matcher(use_sgbm)

        print(f"  Baseline:   {self._baseline_mm:.1f} mm")
        print(f"  Resolution: {w}x{h}")
        print(f"  Matcher:    {'SGBM' if use_sgbm else 'BM'}")
        print(f"  Cameras:    left={cam_left} right={cam_right}")
        print("Stereo ready.")

    # ------------------------------------------------------------------
    # Matcher
    # ------------------------------------------------------------------

    def _build_matcher(self, sgbm: bool):
        if sgbm:
            return cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=96,
                blockSize=5,
                P1=8 * 3 * 5 ** 2,
                P2=32 * 3 * 5 ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        else:
            bm = cv2.StereoBM_create(numDisparities=96, blockSize=15)
            bm.setUniquenessRatio(10)
            bm.setSpeckleWindowSize(100)
            bm.setSpeckleRange(32)
            return bm

    def tune(self, num_disparities: int = 96, block_size: int = 5):
        """Adjust matcher parameters without rebuilding."""
        self._matcher.setNumDisparities(num_disparities)
        self._matcher.setBlockSize(block_size)
        if self._use_sgbm:
            self._matcher.setP1(8 * 3 * block_size ** 2)
            self._matcher.setP2(32 * 3 * block_size ** 2)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if not self._running:
            self._cam_l.start()
            self._cam_r.start()
            self._running = True

    def stop(self):
        if self._running:
            self._cam_l.stop()
            self._cam_r.stop()
            self._running = False

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_raw(self) -> tuple[np.ndarray, np.ndarray]:
        left  = self._cam_l.capture_array()
        right = self._cam_r.capture_array()
        if self._flip:
            left  = cv2.flip(left,  -1)
            right = cv2.flip(right, -1)
        return left, right

    def rectify(self, left, right):
        left_r  = cv2.remap(left,  self._map_lx, self._map_ly, cv2.INTER_LINEAR)
        right_r = cv2.remap(right, self._map_rx, self._map_ry, cv2.INTER_LINEAR)
        return left_r, right_r

    def capture(self) -> StereoFrame:
        t0 = time.perf_counter()
        left_raw, right_raw = self.capture_raw()
        left_rect, right_rect = self.rectify(left_raw, right_raw)
        capture_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        left_gray  = cv2.cvtColor(left_rect,  cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_RGB2GRAY)

        disp_raw = self._matcher.compute(left_gray, right_gray)
        disp = disp_raw.astype(np.float32) / 16.0

        # No smoothing for now - use raw disparity directly
        # Re-enable once depth is confirmed working
        valid = disp >= 1.0
        disparity_ms = (time.perf_counter() - t1) * 1000

        # Metric depth (T is in mm since square_mm was in mm)
        pts = cv2.reprojectImageTo3D(disp, self._Q)
        depth_mm = pts[:, :, 2].copy()
        invalid = disp < 1.0
        depth_mm[invalid] = 0

        # Colorize
        disp_vis  = np.clip(disp, 0, 96)
        disp_norm = (disp_vis / 96 * 255).astype(np.uint8)
        colorized = cv2.applyColorMap(disp_norm, cv2.COLORMAP_MAGMA)
        colorized[invalid] = 0

        return StereoFrame(
            left=left_rect,
            right=right_rect,
            disparity=disp,
            depth_mm=depth_mm,
            colorized=colorized,
            capture_ms=capture_ms,
            disparity_ms=disparity_ms,
        )

    # ------------------------------------------------------------------
    # Obstacle helpers
    # ------------------------------------------------------------------

    def get_obstacle_distances_mm(
        self,
        depth_mm: np.ndarray,
        regions: int = 3,
        min_valid_mm: float = 100,
        max_valid_mm: float = 5000,
    ) -> list[Optional[float]]:
        h, w = depth_mm.shape
        rw = w // regions
        results = []
        for i in range(regions):
            x0 = i * rw
            x1 = (i + 1) * rw if i < regions - 1 else w
            region = depth_mm[:, x0:x1]
            valid = region[(region >= min_valid_mm) & (region <= max_valid_mm)]
            results.append(float(np.median(valid)) if valid.size > 0 else None)
        return results

    def check_path_clear(
        self,
        depth_mm: np.ndarray,
        obstacle_threshold_mm: float = 600,
    ) -> tuple[bool, str]:
        left_d, center_d, right_d = self.get_obstacle_distances_mm(depth_mm)

        def safe(d): return d is None or d > obstacle_threshold_mm

        if safe(center_d):   return True,  'forward'
        if safe(left_d):     return False, 'left'
        if safe(right_d):    return False, 'right'
        return False, 'reverse'

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def create_debug_view(self, frame: StereoFrame) -> np.ndarray:
        h, w = frame.left.shape[:2]
        overlay = frame.left.copy()
        dists = self.get_obstacle_distances_mm(frame.depth_mm)
        labels = ['L', 'C', 'R']
        rw = w // 3

        for i, (lbl, d) in enumerate(zip(labels, dists)):
            x = 10 + i * rw
            if d is not None:
                color = (0, 255, 0) if d > 600 else (0, 165, 255) if d > 300 else (0, 0, 255)
                cv2.putText(overlay, f'{lbl}:{d/1000:.2f}m',
                            (x, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.putText(overlay, f'{lbl}:--',
                            (x, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        cv2.putText(overlay,
                    f'cap:{frame.capture_ms:.0f}ms disp:{frame.disparity_ms:.0f}ms',
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return np.hstack([overlay, frame.colorized])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def run_calibration(
    cam_left: int = 0,
    cam_right: int = 1,
    resolution: tuple[int, int] = (640, 480),
    board_size: tuple[int, int] = (9, 8),
    square_mm: float = 19.0,
    n_frames: int = 30,
    output: str = 'stereo_calib.npz',
    flip: bool = True,
):
    print(f"Stereo calibration: need {n_frames} good stereo pairs")
    print("Tap CAPTURE button or press SPACE | q = finish (>=10 pairs) | ESC = abort\n")

    cam_l = Picamera2(cam_left)
    cam_r = Picamera2(cam_right)
    for cam in (cam_l, cam_r):
        cfg = cam.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        cam.configure(cfg)
        cam.start()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    bw, bh = board_size
    obj_pts_single = np.zeros((bw * bh, 3), np.float32)
    obj_pts_single[:, :2] = np.mgrid[0:bw, 0:bh].T.reshape(-1, 2) * square_mm

    obj_pts, img_pts_l, img_pts_r = [], [], []
    captured = 0
    w, h = resolution
    btn_clicked = False
    corners_state = {"ok_l": False, "ok_r": False, "cl": None, "cr": None, "gl": None, "gr": None}
    state_lock = threading.Lock()
    detection_busy = False

    def detect_corners(gl, gr):
        nonlocal detection_busy
        ok_l, cl = cv2.findChessboardCorners(gl, board_size)
        ok_r, cr = cv2.findChessboardCorners(gr, board_size)
        with state_lock:
            corners_state.update(ok_l=ok_l, ok_r=ok_r, cl=cl, cr=cr, gl=gl, gr=gr)
        detection_busy = False

    def on_mouse(event, x, y, flags, param):
        nonlocal btn_clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if (w - 120) <= x <= (w + 120) and (h - 60) <= y <= (h - 10):
                btn_clicked = True

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("Calibration", on_mouse)

    img_dir = Path("calib_images")
    img_dir.mkdir(exist_ok=True)

    try:
        while True:
            left  = cam_l.capture_array()
            right = cam_r.capture_array()
            if flip:
                left  = cv2.flip(left,  -1)
                right = cv2.flip(right, -1)

            gl = cv2.cvtColor(left,  cv2.COLOR_RGB2GRAY)
            gr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

            if not detection_busy:
                detection_busy = True
                threading.Thread(target=detect_corners, args=(gl.copy(), gr.copy()), daemon=True).start()

            with state_lock:
                ok_l, ok_r = corners_state["ok_l"], corners_state["ok_r"]
                cl, cr     = corners_state["cl"],   corners_state["cr"]
                det_gl     = corners_state["gl"]
                det_gr     = corners_state["gr"]

            both_found = ok_l and ok_r
            vis_l = cv2.cvtColor(gl, cv2.COLOR_GRAY2BGR)
            vis_r = cv2.cvtColor(gr, cv2.COLOR_GRAY2BGR)
            if ok_l and cl is not None: cv2.drawChessboardCorners(vis_l, board_size, cl, ok_l)
            if ok_r and cr is not None: cv2.drawChessboardCorners(vis_r, board_size, cr, ok_r)

            status = f"BOTH FOUND ({captured}/{n_frames})" if both_found else f"Searching... ({captured}/{n_frames})"
            cv2.putText(vis_l, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if both_found else (0, 100, 255), 2)

            combined = np.hstack([vis_l, vis_r])
            btn_color = (0, 180, 0) if both_found else (60, 60, 60)
            cv2.rectangle(combined, (w - 120, h - 60), (w + 120, h - 10), btn_color, -1)
            cv2.rectangle(combined, (w - 120, h - 60), (w + 120, h - 10), (255, 255, 255), 2)
            cv2.putText(combined, "CAPTURE" if both_found else "WAITING...",
                        (w - 78, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Calibration", combined)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                print("Aborted.")
                return

            if ((btn_clicked and both_found) or (key == ord(' ') and both_found)) and det_gl is not None:
                btn_clicked = False
                c_l = cv2.cornerSubPix(det_gl, cl, (11, 11), (-1, -1), criteria)
                c_r = cv2.cornerSubPix(det_gr, cr, (11, 11), (-1, -1), criteria)
                obj_pts.append(obj_pts_single)
                img_pts_l.append(c_l)
                img_pts_r.append(c_r)
                captured += 1
                cv2.imwrite(str(img_dir / f"left_{captured:02d}.jpg"),  cv2.cvtColor(left,  cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(img_dir / f"right_{captured:02d}.jpg"), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
                print(f"  Captured pair {captured}/{n_frames}")
                if captured >= n_frames:
                    break
            else:
                btn_clicked = False

            if key == ord('q') and captured >= 10:
                break

    finally:
        cam_l.stop(); cam_r.stop()
        cv2.destroyAllWindows()

    if captured < 10:
        print(f"Not enough pairs ({captured}), need at least 10.")
        return

    _compute_calibration(obj_pts, img_pts_l, img_pts_r, resolution, output)


def recalibrate_from_images(
    image_dir: str = 'calib_images',
    board_size: tuple[int, int] = (9, 8),
    square_mm: float = 19.0,
    output: str = 'stereo_calib.npz',
):
    """Recompute calibration from saved images without cameras."""
    img_dir = Path(image_dir)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    bw, bh = board_size
    obj_pts_single = np.zeros((bw * bh, 3), np.float32)
    obj_pts_single[:, :2] = np.mgrid[0:bw, 0:bh].T.reshape(-1, 2) * square_mm

    obj_pts, img_pts_l, img_pts_r = [], [], []
    img_size = None

    pairs = sorted(glob.glob(str(img_dir / 'left_*.jpg')))
    print(f"Found {len(pairs)} image pairs in {image_dir}/")

    for left_path in pairs:
        right_path = left_path.replace('left_', 'right_')
        if not Path(right_path).exists():
            print(f"  Missing: {right_path}")
            continue

        left  = cv2.imread(left_path,  cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        if img_size is None:
            img_size = (left.shape[1], left.shape[0])

        ok_l, cl = cv2.findChessboardCorners(left,  board_size)
        ok_r, cr = cv2.findChessboardCorners(right, board_size)

        if ok_l and ok_r:
            cl = cv2.cornerSubPix(left,  cl, (11, 11), (-1, -1), criteria)
            cr = cv2.cornerSubPix(right, cr, (11, 11), (-1, -1), criteria)
            obj_pts.append(obj_pts_single)
            img_pts_l.append(cl)
            img_pts_r.append(cr)
            print(f"  OK:   {Path(left_path).name}")
        else:
            print(f"  SKIP: {Path(left_path).name}")

    print(f"\nUsing {len(obj_pts)} valid pairs...")
    _compute_calibration(obj_pts, img_pts_l, img_pts_r, img_size, output)


def _compute_calibration(obj_pts, img_pts_l, img_pts_r, resolution, output):
    """Shared calibration computation used by both calibration paths."""
    w, h = resolution if isinstance(resolution, tuple) else (resolution[0], resolution[1])

    _, K1, D1, _, _ = cv2.calibrateCamera(obj_pts, img_pts_l, (w, h), None, None)
    _, K2, D2, _, _ = cv2.calibrateCamera(obj_pts, img_pts_r, (w, h), None, None)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts_l, img_pts_r,
        K1, D1, K2, D2, (w, h),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=criteria
    )
    print(f"  RMS reprojection error: {rms:.4f} px  (target < 0.5)")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    np.savez(output, K1=K1, D1=D1, K2=K2, D2=D2,
             R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
    print(f"  Saved to {output}")
    print(f"  Baseline: {abs(float(T[0].item())):.1f} mm")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import os
    os.environ['DISPLAY'] = ':0'

    stereo = StereoCamera(cam_left=1, cam_right=0)
    stereo.start()

    cv2.namedWindow("Stereo", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Stereo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            frame = stereo.capture()
            clear, action = stereo.check_path_clear(frame.depth_mm)
            vis = stereo.create_debug_view(frame)
            label = f"{'CLEAR' if clear else 'BLOCKED'} -> {action.upper()}"
            cv2.putText(vis, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Stereo", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stereo.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if args and args[0].lstrip('-') == 'calibrate':
        run_calibration()
    elif args and args[0].lstrip('-') == 'recalibrate':
        recalibrate_from_images()
    else:
        demo()