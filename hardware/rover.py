"""
hardware/rover.py — RoverCore

Single entry point for all rover hardware. Owns motors, IMU, and depth fusion.
High-level scripts just instantiate this and call methods — nothing else to think about.

Built-in automatically (no configuration needed at script level):
  - Emergency stop on SIGINT, SIGTERM, SIGHUP, atexit
  - IMU throttled to 5Hz (every 10 control cycles at 50Hz)
  - Depth fusion capped at 10fps
  - Feedforward drift correction from config.yaml
  - Speed floor enforced at MIN_SPEED_FLOOR (400 tps on carpet)
  - Acceleration ramp from floor to target speed

Usage:
    from hardware.rover import RoverCore

    rover = RoverCore()
    rover.start()
    rover.drive_distance(500)
    rover.pivot_turn(90)
    rover.drive_distance(1000)
    rover.shutdown()
"""
import os
import time
import signal
import atexit
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml

import cv2
from mpu9250_jmdev.registers import *
from mpu9250_jmdev.mpu_9250 import MPU9250

from hardware.motor_controller import MotorDriver, TICKS_PER_WHEEL_REV
from depth_fusion_new import DepthFusion

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).parent.parent
CONFIG_PATH = ROOT / 'config.yaml'
with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

WHEEL_DIAMETER_MM    = _cfg['robot_geometry']['wheel_diameter_mm']
WHEEL_BASE_MM        = _cfg['robot_geometry']['wheel_base_mm']
WHEEL_CIRCUMFERENCE_MM = WHEEL_DIAMETER_MM * np.pi
MM_PER_TICK          = WHEEL_CIRCUMFERENCE_MM / TICKS_PER_WHEEL_REV

MIN_SPEED_FLOOR = 400   # tps — carpet stiction threshold
IMU_READ_EVERY  = 10    # gyro read every N cycles → 5Hz at 50Hz loop
DEPTH_FPS       = 10    # max depth fusion framerate
CONTROL_HZ      = 50    # nav control loop rate


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoverState:
    timestamp:   float
    x_mm:        float
    y_mm:        float
    heading_deg: float
    speed_left:  float
    speed_right: float


# ---------------------------------------------------------------------------
# RoverCore
# ---------------------------------------------------------------------------

class RoverCore:
    """
    Central coordinator for all rover hardware and control loops.
    Instantiate once, call start(), use navigation methods, call shutdown().
    """

    def __init__(
        self,
        gyro_calibration_samples: int = 200,
        enable_depth: bool = True,
        depth_output: str = 'depth_recording.avi',
    ):
        print("Initializing RoverCore...")

        # Register emergency stop before touching any hardware
        self._register_emergency_stop()

        # --- Motors ---
        self._driver = MotorDriver()

        # --- IMU ---
        self._mpu = MPU9250(
            address_ak=AK8963_ADDRESS,
            address_mpu_master=MPU9050_ADDRESS_68,
            address_mpu_slave=None,
            bus=1, gfs=GFS_1000, afs=AFS_8G,
            mfs=AK8963_BIT_16, mode=AK8963_MODE_C100HZ
        )
        self._mpu.configure()
        print("  Calibrating gyro... keep still")
        readings = [self._mpu.readGyroscopeMaster()[2]
                    for _ in range(gyro_calibration_samples)
                    if not time.sleep(0.01)]
        self._gyro_bias = np.mean(readings)
        print(f"  Gyro bias: {self._gyro_bias:.4f}°/s")

        # --- Drift correction table ---
        raw = _cfg.get('drift_correction', {})
        self._drift_table = {int(k): float(v) for k, v in raw.items()}
        if self._drift_table:
            print(f"  Drift correction loaded: {sorted(self._drift_table.keys())} tps")
        else:
            print("  No drift correction — run tools/calibrate_drift.py first")

        # --- Depth fusion (optional) ---
        self._depth_enabled = enable_depth
        self._fusion: Optional[DepthFusion] = None
        self._depth_thread: Optional[threading.Thread] = None
        self._depth_running = False
        self._depth_period  = 1.0 / DEPTH_FPS
        self._latest_frame  = None
        self._frame_lock    = threading.Lock()
        self._depth_writer  = None

        if enable_depth:
            print("  Loading depth fusion...")
            self._fusion = DepthFusion()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self._depth_writer = cv2.VideoWriter(
                depth_output, fourcc, float(DEPTH_FPS), (1280, 480)
            )
            print(f"  Recording depth to: {depth_output}")

        # --- Nav state ---
        self._x = 0.0
        self._y = 0.0
        self._heading    = 0.0
        self._last_left_ticks  = 0
        self._last_right_ticks = 0
        self._last_gz    = 0.0
        self._imu_cycle  = 0

        self._target_speed  = 0.0
        self._current_speed = 0.0
        self._accel_rate    = 400  # tps/s
        self._correction_enabled = True

        # Turn state
        self._turning         = False
        self._turn_target     = 0.0
        self._turn_speed      = 0.0
        self._turn_radius     = 0.0
        self._turn_direction  = 1
        self._turn_complete   = False
        self._turn_start_time = 0.0

        # PID
        self._heading_integral   = 0.0
        self._lateral_integral   = 0.0
        self._last_heading_error = 0.0

        self.heading_kp = 5.0
        self.heading_ki = 1.5
        self.heading_kd = 1.2
        self.lateral_kp = 0.4
        self.lateral_ki = 0.02
        self.max_correction         = 250
        self.max_heading_correction = 45

        # Threading
        self._running        = False
        self._control_thread: Optional[threading.Thread] = None
        self._telemetry      = deque(maxlen=2000)

        print("RoverCore ready.")

    # ------------------------------------------------------------------
    # Emergency stop — lowest level, registered before any hardware init
    # ------------------------------------------------------------------

    def _register_emergency_stop(self):
        def _stop(signum=None, frame=None):
            try:
                self._driver.left.write_pwm(0)
                self._driver.right.write_pwm(0)
                self._driver.left.en.off()
                self._driver.right.en.off()
            except Exception:
                pass
            if signum is not None:
                signal.signal(signum, signal.SIG_DFL)
                signal.raise_signal(signum)

        atexit.register(_stop)
        signal.signal(signal.SIGINT,  _stop)
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGHUP,  _stop)

    # ------------------------------------------------------------------
    # Drift correction
    # ------------------------------------------------------------------

    def _get_drift_correction(self, speed: float) -> float:
        if not self._drift_table:
            return 0.0
        speeds = sorted(self._drift_table.keys())
        if speed <= speeds[0]:  return self._drift_table[speeds[0]]
        if speed >= speeds[-1]: return self._drift_table[speeds[-1]]
        for i in range(len(speeds) - 1):
            lo, hi = speeds[i], speeds[i + 1]
            if lo <= speed <= hi:
                t = (speed - lo) / (hi - lo)
                return self._drift_table[lo] + t * (self._drift_table[hi] - self._drift_table[lo])
        return 0.0

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def _read_gyro(self) -> float:
        _, _, gz = self._mpu.readGyroscopeMaster()
        gz = -(gz - self._gyro_bias)
        return gz if abs(gz) > 0.5 else 0.0

    def _update_position(self, dt: float):
        lt = self._driver.left.ticks
        rt = self._driver.right.ticks
        dl = (lt - self._last_left_ticks)  * MM_PER_TICK
        dr = -(rt - self._last_right_ticks) * MM_PER_TICK
        self._last_left_ticks  = lt
        self._last_right_ticks = rt
        dist = (dl + dr) / 2.0
        hr = np.radians(self._heading)
        self._x += dist * np.cos(hr)
        self._y += dist * np.sin(hr)

    # ------------------------------------------------------------------
    # PID
    # ------------------------------------------------------------------

    def _compute_correction(self, dt: float) -> float:
        if not self._correction_enabled:
            return 0.0
        self._lateral_integral = np.clip(
            self._lateral_integral + self._y * dt, -500, 500)
        target_heading = np.clip(
            -(self.lateral_kp * self._y + self.lateral_ki * self._lateral_integral),
            -self.max_heading_correction, self.max_heading_correction)
        err = self._heading - target_heading
        self._heading_integral = np.clip(
            self._heading_integral + err * dt, -100, 100)
        d = (err - self._last_heading_error) / dt if dt > 0 else 0
        self._last_heading_error = err
        return np.clip(
            self.heading_kp * err + self.heading_ki * self._heading_integral + self.heading_kd * d,
            -self.max_correction, self.max_correction)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self):
        period    = 1.0 / CONTROL_HZ
        last_time = time.monotonic()

        while self._running:
            now = time.monotonic()
            dt  = now - last_time
            last_time = now

            # Acceleration ramp
            if self._current_speed < self._target_speed:
                self._current_speed = min(self._current_speed + self._accel_rate * dt, self._target_speed)
            elif self._current_speed > self._target_speed:
                self._current_speed = max(self._current_speed - self._accel_rate * dt, self._target_speed)

            # IMU — throttled to 5Hz
            self._imu_cycle += 1
            if self._imu_cycle >= IMU_READ_EVERY:
                self._last_gz   = self._read_gyro()
                self._imu_cycle = 0
            self._heading += self._last_gz * dt
            self._update_position(dt)

            if self._turning:
                self._run_turn(now)
            elif self._current_speed > 0:
                pid   = self._compute_correction(dt)
                spd   = self._current_speed
                drift = self._get_drift_correction(spd)
                l = max(MIN_SPEED_FLOOR, spd - pid - drift / 2)
                r = max(MIN_SPEED_FLOOR, spd + pid + drift / 2)
                self._driver.set_speed_tps(l, r)

            self._telemetry.append(RoverState(
                timestamp=now, x_mm=self._x, y_mm=self._y,
                heading_deg=self._heading,
                speed_left=self._driver.left.speed_tps,
                speed_right=self._driver.right.speed_tps,
            ))

            elapsed = time.monotonic() - now
            if elapsed < period:
                time.sleep(period - elapsed)

    # ------------------------------------------------------------------
    # Depth loop
    # ------------------------------------------------------------------

    def _depth_loop(self):
        if self._fusion is None:
            return
        cv2.namedWindow("Depth Fusion", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Depth Fusion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while self._depth_running:
            t0    = time.monotonic()
            frame = self._fusion.capture()
            vis   = self._fusion.create_debug_view(frame)

            with self._frame_lock:
                self._latest_frame = frame

            cv2.imshow("Depth Fusion", vis)
            if self._depth_writer:
                self._depth_writer.write(vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.monotonic() - t0
            if elapsed < self._depth_period:
                time.sleep(self._depth_period - elapsed)

        cv2.destroyAllWindows()
        if self._depth_writer:
            self._depth_writer.release()

    # ------------------------------------------------------------------
    # Turn helpers
    # ------------------------------------------------------------------

    def _run_turn(self, now: float):
        if now - self._turn_start_time > 10.0:
            print("[SAFETY] Turn timeout!")
            self._turning = False
            self._driver.stop()
            return

        done = (self._heading >= self._turn_target if self._turn_direction > 0
                else self._heading <= self._turn_target)
        if done:
            self._turning = False
            self._turn_complete = True
            self._driver.stop()
            return

        remaining    = abs(self._turn_target - self._heading)
        speed_factor = max(0.5, min(1.0, remaining / 15))
        l, r = self._compute_turn_speeds()
        l *= speed_factor
        r *= speed_factor

        if self._turn_radius == 0:
            self._driver.left.write_pwm( np.clip(abs(l) / 1000 * np.sign(l), -0.8, 0.8))
            self._driver.right.write_pwm(np.clip(abs(r) / 1000 * np.sign(r), -0.8, 0.8))
        elif self._turn_radius == float('inf'):
            self._driver.left.write_pwm( min(0.9, abs(l) / 800) if l > 0 else 0)
            self._driver.right.write_pwm(min(0.9, abs(r) / 800) if r > 0 else 0)
        else:
            self._driver.set_speed_tps(max(0, l), max(0, r))

    def _compute_turn_speeds(self):
        if self._turn_radius == 0:
            return (self._turn_speed * self._turn_direction,
                   -self._turn_speed * self._turn_direction)
        if self._turn_radius == float('inf'):
            return ((self._turn_speed, 0) if self._turn_direction > 0
                    else (0, self._turn_speed))
        r, w = self._turn_radius, WHEEL_BASE_MM / 2
        if self._turn_direction > 0:
            return self._turn_speed * (r + w) / r, self._turn_speed * (r - w) / r
        return self._turn_speed * (r - w) / r, self._turn_speed * (r + w) / r

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, warmup: float = 3.0):
        self._driver.start()
        if self._fusion:
            self._fusion.start()

        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

        if self._fusion:
            self._depth_running = True
            self._depth_thread  = threading.Thread(target=self._depth_loop, daemon=True)
            self._depth_thread.start()
            if warmup > 0:
                print(f"  Warming up depth fusion ({warmup:.0f}s)...")
                time.sleep(warmup)

    def shutdown(self):
        self._running       = False
        self._depth_running = False
        self._turning       = False
        self._target_speed  = 0.0
        self._current_speed = 0.0
        self._driver.left.write_pwm(0)
        self._driver.right.write_pwm(0)
        self._driver.stop()
        if self._control_thread:
            self._control_thread.join(timeout=1.0)
        if self._depth_thread:
            self._depth_thread.join(timeout=3.0)
        if self._fusion:
            self._fusion.stop()
        self._driver.disable()
        print("RoverCore shut down.")

    # ------------------------------------------------------------------
    # Navigation API
    # ------------------------------------------------------------------

    def stop(self):
        self._target_speed  = 0.0
        self._current_speed = 0.0
        self._turning       = False
        self._driver.left.write_pwm(0)
        self._driver.right.write_pwm(0)
        self._driver.stop()

    def drive(self, speed_tps: float, reset_position: bool = True):
        if reset_position:
            self.reset_tracking()
        self._target_speed  = abs(speed_tps)
        self._current_speed = MIN_SPEED_FLOOR

    def drive_distance(self, distance_mm: float, speed_tps: float = 600) -> bool:
        self.reset_tracking()
        self.drive(speed_tps, reset_position=False)
        target = abs(distance_mm)

        while self._running:
            current = abs(self._x)
            if current >= target:
                self.stop()
                return True
            remaining = target - current
            if remaining < 80:
                self._target_speed = max(MIN_SPEED_FLOOR, speed_tps * (remaining / 80))
            time.sleep(0.05)
        return False

    def pivot_turn(self, degrees: float, speed_tps: float = 300):
        self._turn_target     = self._heading + degrees
        self._turn_speed      = speed_tps
        self._turn_radius     = 0
        self._turn_direction  = 1 if degrees > 0 else -1
        self._turn_complete   = False
        self._turn_start_time = time.monotonic()
        self._turning         = True

    def swing_turn(self, degrees: float, speed_tps: float = 300):
        self._turn_target     = self._heading + degrees
        self._turn_speed      = speed_tps
        self._turn_radius     = float('inf')
        self._turn_direction  = 1 if degrees > 0 else -1
        self._turn_complete   = False
        self._turn_start_time = time.monotonic()
        self._turning         = True

    def arc_turn(self, degrees: float, radius_mm: float, speed_tps: float = 400):
        min_r = WHEEL_BASE_MM / 2 + 10
        radius_mm = max(radius_mm, min_r)
        self._turn_target     = self._heading + degrees
        self._turn_speed      = speed_tps
        self._turn_radius     = radius_mm
        self._turn_direction  = 1 if degrees > 0 else -1
        self._turn_complete   = False
        self._turn_start_time = time.monotonic()
        self._turning         = True

    def wait_for_turn(self, timeout: float = 10.0) -> bool:
        start = time.monotonic()
        while self._turning and (time.monotonic() - start) < timeout:
            time.sleep(0.02)
        return self._turn_complete

    def drive_to(self, x_mm: float, y_mm: float, speed_tps: float = 600) -> bool:
        dx, dy = x_mm - self._x, y_mm - self._y
        angle  = np.degrees(np.arctan2(dy, dx))
        dist   = np.hypot(dx, dy)
        diff   = (angle - self._heading + 180) % 360 - 180
        if abs(diff) > 2:
            self.pivot_turn(diff)
            if not self.wait_for_turn():
                return False
            time.sleep(0.2)
        return self.drive_distance(dist, speed_tps)

    def drive_path(self, waypoints: list[tuple[float, float]], speed_tps: float = 600) -> bool:
        for x, y in waypoints:
            if not self.drive_to(x, y, speed_tps):
                return False
            time.sleep(0.1)
        return True

    def reset_tracking(self):
        self._x = self._y = self._heading = 0.0
        self._heading_integral   = 0.0
        self._lateral_integral   = 0.0
        self._last_heading_error = 0.0
        self._last_left_ticks    = self._driver.left.ticks
        self._last_right_ticks   = self._driver.right.ticks

    def enable_correction(self, enabled: bool = True):
        self._correction_enabled = enabled
        if enabled:
            self._heading_integral = 0.0
            self._lateral_integral = 0.0

    # ------------------------------------------------------------------
    # Depth API
    # ------------------------------------------------------------------

    @property
    def latest_frame(self):
        with self._frame_lock:
            return self._latest_frame

    def path_clear(self, threshold_mm: float = 600) -> tuple[bool, str]:
        frame = self.latest_frame
        if frame is None:
            return True, 'unknown'
        return self._fusion.check_path_clear(frame.depth_metric, threshold_mm)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> tuple[float, float]:
        return self._x, self._y

    @property
    def heading(self) -> float:
        return self._heading

    @property
    def is_turning(self) -> bool:
        return self._turning

    @property
    def state(self) -> RoverState:
        return self._telemetry[-1] if self._telemetry else RoverState(time.monotonic(), 0, 0, 0, 0, 0)