"""
brain/navigation.py: Plans/coordinates motor movements.
Changes:
  - IMU only read every N control cycles (default every 5 = 10Hz effective)
  - Feedforward drift correction loaded from config at startup
  - Acceleration ramp always starts at MIN_SPEED_FLOOR (200 tps)
"""
import time
import threading
import numpy as np
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import yaml

from mpu9250_jmdev.registers import *
from mpu9250_jmdev.mpu_9250 import MPU9250
from hardware.motor_controller import MotorDriver, TICKS_PER_WHEEL_REV

current_folder = Path(__file__).parent
root_folder    = current_folder.parent
config_path    = root_folder / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

WHEEL_DIAMETER_MM    = config['robot_geometry']['wheel_diameter_mm']
WHEEL_BASE_MM        = config['robot_geometry']['wheel_base_mm']
WHEEL_CIRCUMFERENCE_MM = WHEEL_DIAMETER_MM * np.pi
MM_PER_TICK          = WHEEL_CIRCUMFERENCE_MM / TICKS_PER_WHEEL_REV

MIN_SPEED_FLOOR = 400  # tps — carpet requires higher stiction threshold
IMU_READ_EVERY  = 10   # read gyro every N control cycles (~5Hz at 50Hz loop)


@dataclass
class RobotState:
    timestamp:   float
    x_mm:        float
    y_mm:        float
    heading_deg: float
    speed_left:  float
    speed_right: float


class TrackController:
    def __init__(self, gyro_calibration_samples: int = 200):
        print("Initializing TrackController...")

        self._driver = MotorDriver()
        self._mpu = MPU9250(
            address_ak=AK8963_ADDRESS,
            address_mpu_master=MPU9050_ADDRESS_68,
            address_mpu_slave=None,
            bus=1, gfs=GFS_1000, afs=AFS_8G,
            mfs=AK8963_BIT_16, mode=AK8963_MODE_C100HZ
        )
        self._mpu.configure()

        # Gyro calibration
        print("  Calibrating gyro... keep still")
        readings = [self._mpu.readGyroscopeMaster()[2]
                    for _ in range(gyro_calibration_samples)
                    if not time.sleep(0.01)]
        self._gyro_bias = np.mean(readings)
        print(f"  Gyro bias: {self._gyro_bias:.4f}°/s")

        # Load feedforward drift correction table from config
        # Format: {speed_tps: correction_tps} — positive = left faster
        raw = config.get('drift_correction', {})
        self._drift_table = {int(k): float(v) for k, v in raw.items()}
        if self._drift_table:
            print(f"  Drift correction loaded for speeds: {sorted(self._drift_table.keys())}")
        else:
            print("  No drift correction found — run tools/calibrate_drift.py first")

        # State
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._last_left_ticks  = 0
        self._last_right_ticks = 0
        self._last_gz = 0.0        # last known gyro reading (reused between IMU reads)
        self._imu_cycle = 0        # counts control cycles for IMU throttle

        # Control targets
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

        # PID state
        self._heading_integral   = 0.0
        self._lateral_integral   = 0.0
        self._last_heading_error = 0.0

        # Tuning
        self.heading_kp = 5.0
        self.heading_ki = 1.5
        self.heading_kd = 1.2
        self.lateral_kp = 0.4
        self.lateral_ki = 0.02
        self.max_correction         = 250
        self.max_heading_correction = 45

        # Threading
        self._running        = False
        self._control_thread = None
        self._telemetry      = deque(maxlen=2000)
        self._control_rate   = 50  # Hz

        print("TrackController ready.")

    # ------------------------------------------------------------------
    # Drift correction helpers
    # ------------------------------------------------------------------

    def _get_drift_correction(self, speed_tps: float) -> float:
        """
        Interpolate drift correction from calibration table.
        Returns tps to subtract from left / add to right.
        """
        if not self._drift_table:
            return 0.0
        speeds = sorted(self._drift_table.keys())
        if speed_tps <= speeds[0]:
            return self._drift_table[speeds[0]]
        if speed_tps >= speeds[-1]:
            return self._drift_table[speeds[-1]]
        # Linear interpolation between bracketing entries
        for i in range(len(speeds) - 1):
            lo, hi = speeds[i], speeds[i + 1]
            if lo <= speed_tps <= hi:
                t = (speed_tps - lo) / (hi - lo)
                return self._drift_table[lo] + t * (self._drift_table[hi] - self._drift_table[lo])
        return 0.0

    def _apply_drift_feedforward(self, left_tps: float, right_tps: float,
                                  speed_tps: float) -> tuple[float, float]:
        """Offset wheel targets by calibrated drift before sending to driver."""
        correction = self._get_drift_correction(speed_tps)
        # positive correction = left was faster = slow left, speed right
        left_tps  -= correction / 2
        right_tps += correction / 2
        return left_tps, right_tps

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def _read_gyro(self) -> float:
        """Read gyro Z with bias correction. Returns 0 below noise floor."""
        _, _, gz = self._mpu.readGyroscopeMaster()
        gz = -(gz - self._gyro_bias)
        return gz if abs(gz) > 0.5 else 0.0

    def _update_position(self, dt: float):
        """Dead-reckoning position update from encoders + heading."""
        left_ticks  = self._driver.left.ticks
        right_ticks = self._driver.right.ticks

        dl = (left_ticks  - self._last_left_ticks)  * MM_PER_TICK
        dr = -(right_ticks - self._last_right_ticks) * MM_PER_TICK

        self._last_left_ticks  = left_ticks
        self._last_right_ticks = right_ticks

        dist = (dl + dr) / 2.0
        heading_rad = np.radians(self._heading)
        self._x += dist * np.cos(heading_rad)
        self._y += dist * np.sin(heading_rad)

    # ------------------------------------------------------------------
    # PID correction
    # ------------------------------------------------------------------

    def _compute_correction(self, dt: float) -> float:
        if not self._correction_enabled:
            return 0.0

        self._lateral_integral += self._y * dt
        self._lateral_integral  = np.clip(self._lateral_integral, -500, 500)

        target_heading = -(self.lateral_kp * self._y + self.lateral_ki * self._lateral_integral)
        target_heading  = np.clip(target_heading, -self.max_heading_correction, self.max_heading_correction)

        heading_error = self._heading - target_heading

        self._heading_integral += heading_error * dt
        self._heading_integral  = np.clip(self._heading_integral, -100, 100)

        d_term = (heading_error - self._last_heading_error) / dt if dt > 0 else 0
        self._last_heading_error = heading_error

        correction = (self.heading_kp * heading_error +
                      self.heading_ki * self._heading_integral +
                      self.heading_kd * d_term)

        return np.clip(correction, -self.max_correction, self.max_correction)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self):
        period    = 1.0 / self._control_rate
        last_time = time.monotonic()

        while self._running:
            now = time.monotonic()
            dt  = now - last_time
            last_time = now

            # Acceleration ramp — floor at MIN_SPEED_FLOOR while moving
            if self._current_speed < self._target_speed:
                self._current_speed = min(self._current_speed + self._accel_rate * dt,
                                          self._target_speed)
            elif self._current_speed > self._target_speed:
                self._current_speed = max(self._current_speed - self._accel_rate * dt,
                                          self._target_speed)

            # IMU throttle — only read hardware every IMU_READ_EVERY cycles
            self._imu_cycle += 1
            if self._imu_cycle >= IMU_READ_EVERY:
                self._last_gz   = self._read_gyro()
                self._imu_cycle = 0

            self._heading += self._last_gz * dt
            self._update_position(dt)

            if self._turning:
                self._run_turn(now, dt)
            elif self._current_speed > 0:
                pid = self._compute_correction(dt)
                spd = self._current_speed

                # Base targets with PID correction
                l_target = max(MIN_SPEED_FLOOR, spd - pid)
                r_target = max(MIN_SPEED_FLOOR, spd + pid)

                # Apply feedforward drift offset on top
                l_target, r_target = self._apply_drift_feedforward(l_target, r_target, spd)

                self._driver.set_speed_tps(l_target, r_target)

            self._telemetry.append(RobotState(
                timestamp=now,
                x_mm=self._x, y_mm=self._y,
                heading_deg=self._heading,
                speed_left=self._driver.left.speed_tps,
                speed_right=self._driver.right.speed_tps
            ))

            elapsed = time.monotonic() - now
            if elapsed < period:
                time.sleep(period - elapsed)

    def _run_turn(self, now: float, dt: float):
        """Turn logic extracted for readability."""
        if now - self._turn_start_time > 10.0:
            print("\n[SAFETY] Turn timeout!")
            self._turning = False
            self._driver.stop()
            return

        if self._turn_direction > 0:
            done = self._heading >= self._turn_target
        else:
            done = self._heading <= self._turn_target

        if done:
            self._turning       = False
            self._turn_complete = True
            self._driver.stop()
            return

        remaining    = abs(self._turn_target - self._heading)
        speed_factor = max(0.5, min(1.0, remaining / 15))

        left, right = self._compute_turn_speeds()
        left  *= speed_factor
        right *= speed_factor

        if self._turn_radius == 0:
            lp = np.clip((abs(left)  / 1000) * (1 if left  >= 0 else -1), -0.8, 0.8)
            rp = np.clip((abs(right) / 1000) * (1 if right >= 0 else -1), -0.8, 0.8)
            self._driver.left.write_pwm(lp)
            self._driver.right.write_pwm(rp)
        elif self._turn_radius == float('inf'):
            lp = min(0.9, abs(left)  / 800) if left  > 0 else 0
            rp = min(0.9, abs(right) / 800) if right > 0 else 0
            self._driver.left.write_pwm(lp)
            self._driver.right.write_pwm(rp)
        else:
            self._driver.set_speed_tps(max(0, left), max(0, right))

    def _compute_turn_speeds(self) -> tuple[float, float]:
        if self._turn_radius == 0:
            left  =  self._turn_speed * self._turn_direction
            right = -self._turn_speed * self._turn_direction
        elif self._turn_radius == float('inf'):
            left  = self._turn_speed if self._turn_direction > 0 else 0
            right = self._turn_speed if self._turn_direction < 0 else 0
        else:
            r, w = self._turn_radius, WHEEL_BASE_MM / 2
            if self._turn_direction > 0:
                left  = self._turn_speed * (r + w) / r
                right = self._turn_speed * (r - w) / r
            else:
                left  = self._turn_speed * (r - w) / r
                right = self._turn_speed * (r + w) / r
        return left, right

    # ------------------------------------------------------------------
    # Public interface (unchanged API)
    # ------------------------------------------------------------------

    def start(self):
        self._driver.start()
        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

    def drive(self, speed_tps: float, reset_position: bool = True):
        if reset_position:
            self.reset_tracking()
        self._target_speed  = abs(speed_tps)
        self._current_speed = MIN_SPEED_FLOOR  # start at floor, ramp up

    def stop(self):
        self._target_speed  = 0.0
        self._current_speed = 0.0
        self._turning       = False
        self._driver.left.write_pwm(0)
        self._driver.right.write_pwm(0)
        self._driver.stop()

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
        if radius_mm < min_r:
            print(f"Warning: radius clamped to {min_r}mm")
            radius_mm = min_r
        self._turn_target     = self._heading + degrees
        self._turn_speed      = speed_tps
        self._turn_radius     = radius_mm
        self._turn_direction  = 1 if degrees > 0 else -1
        self._turn_complete   = False
        self._turn_start_time = time.monotonic()
        self._turning         = True

    def turn(self, degrees: float, radius_mm: float = 0, speed_tps: float = 300):
        if radius_mm == 0:
            self.pivot_turn(degrees, speed_tps)
        elif radius_mm == float('inf') or radius_mm is None:
            self.swing_turn(degrees, speed_tps)
        else:
            self.arc_turn(degrees, radius_mm, speed_tps)

    def wait_for_turn(self, timeout: float = 10.0) -> bool:
        start = time.monotonic()
        while self._turning and (time.monotonic() - start) < timeout:
            time.sleep(0.02)
        return self._turn_complete

    @property
    def is_turning(self) -> bool:
        return self._turning

    def drive_distance(self, distance_mm: float, speed_tps: float = 600) -> bool:
        self.reset_tracking()
        target_dist = abs(distance_mm)
        self.drive(speed_tps, reset_position=False)

        while self._running:
            current_dist = abs(self._x)
            if current_dist >= target_dist:
                self.stop()
                return True
            remaining = target_dist - current_dist
            if remaining < 50:
                self._target_speed = max(MIN_SPEED_FLOOR, speed_tps * (remaining / 50))
            time.sleep(0.02)
        return False

    def drive_to(self, x_mm: float, y_mm: float, speed_tps: float = 600) -> bool:
        dx = x_mm - self._x
        dy = y_mm - self._y
        target_heading = np.degrees(np.arctan2(dy, dx))
        distance = np.sqrt(dx**2 + dy**2)

        angle_diff = target_heading - self._heading
        while angle_diff >  180: angle_diff -= 360
        while angle_diff < -180: angle_diff += 360

        if abs(angle_diff) > 2:
            self.pivot_turn(angle_diff)
            if not self.wait_for_turn():
                return False
            time.sleep(0.2)
        return self.drive_distance(distance, speed_tps)

    def drive_square(self, size_mm: float, speed_tps: float = 600, turn_type: str = 'pivot') -> bool:
        for _ in range(4):
            if not self.drive_distance(size_mm, speed_tps):
                return False
            time.sleep(0.2)
            {'pivot': self.pivot_turn, 'swing': self.swing_turn}.get(turn_type, self.pivot_turn)(90)
            if not self.wait_for_turn():
                return False
            time.sleep(0.2)
        return True

    def drive_polygon(self, sides: int, size_mm: float, speed_tps: float = 600) -> bool:
        for _ in range(sides):
            if not self.drive_distance(size_mm, speed_tps):
                return False
            time.sleep(0.2)
            self.pivot_turn(360 / sides)
            if not self.wait_for_turn():
                return False
            time.sleep(0.2)
        return True

    def drive_path(self, waypoints: list[tuple[float, float]], speed_tps: float = 600) -> bool:
        for x, y in waypoints:
            if not self.drive_to(x, y, speed_tps):
                return False
            time.sleep(0.1)
        return True

    def shutdown(self):
        self._running       = False
        self._turning       = False
        self._target_speed  = 0.0
        self._current_speed = 0.0
        self._driver.left.write_pwm(0)
        self._driver.right.write_pwm(0)
        self._driver.stop()
        if self._control_thread:
            self._control_thread.join(timeout=0.5)
        self._driver.disable()

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

    @property
    def state(self) -> RobotState:
        return self._telemetry[-1] if self._telemetry else RobotState(time.monotonic(), 0, 0, 0, 0, 0)

    @property
    def position(self) -> tuple[float, float]:
        return (self._x, self._y)

    @property
    def heading(self) -> float:
        return self._heading

    @property
    def telemetry(self) -> list[RobotState]:
        return list(self._telemetry)