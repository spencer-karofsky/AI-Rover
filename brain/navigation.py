"""
brain/navigation.py: Plans/coordinates motor movements
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

# Robot geometry
current_folder = Path(__file__).parent
root_folder = current_folder.parent
config_path = root_folder / 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
WHEEL_DIAMETER_MM = config['robot_geometry']['wheel_diameter_mm']
WHEEL_BASE_MM = config['robot_geometry']['wheel_base_mm']

WHEEL_CIRCUMFERENCE_MM = WHEEL_DIAMETER_MM * np.pi
MM_PER_TICK = WHEEL_CIRCUMFERENCE_MM / TICKS_PER_WHEEL_REV


@dataclass
class RobotState:
    """Current state of the robot."""
    timestamp: float
    x_mm: float # Forward distance from start
    y_mm: float # Lateral offset (positive = right)
    heading_deg: float # Heading (positive = right)
    speed_left: float # Left wheel speed (tps)
    speed_right: float # Right wheel speed (tps)

class TrackController:
    """
    Main controller for straight-line driving with position correction.
    
    Usage:
        controller = TrackController()
        controller.start()
        controller.drive(600)  # Drive at 600 tps
        time.sleep(5)
        controller.stop()
        controller.shutdown()
    """
    def __init__(self, gyro_calibration_samples: int = 200):
        print("Initializing TrackController...")
        
        # Hardware
        self._driver = MotorDriver()
        self._mpu = MPU9250(
            address_ak=AK8963_ADDRESS,
            address_mpu_master=MPU9050_ADDRESS_68,
            address_mpu_slave=None,
            bus=1, gfs=GFS_1000, afs=AFS_8G,
            mfs=AK8963_BIT_16, mode=AK8963_MODE_C100HZ
        )
        self._mpu.configure()
        
        # Calibrate gyro
        print("  Calibrating gyro... keep still")
        readings = [self._mpu.readGyroscopeMaster()[2] for _ in range(gyro_calibration_samples) if not time.sleep(0.01)]
        self._gyro_bias = np.mean(readings)
        print(f"  Gyro bias: {self._gyro_bias:.4f}°/s")
        
        # State
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._last_left_ticks = 0
        self._last_right_ticks = 0
        
        # Control targets
        self._target_speed = 0.0
        self._current_speed = 0.0
        self._accel_rate = 400  # tps per second
        self._correction_enabled = True
        
        # Turn state
        self._turning = False
        self._turn_target = 0.0 # Target heading in degrees
        self._turn_speed = 0.0 # Base speed for turn
        self._turn_radius = 0.0 # Turn radius (0 = pivot)
        self._turn_direction = 1 # 1 = right, -1 = left
        self._turn_complete = False
        self._turn_start_time = 0.0 # For timeout safety
        
        # PID state
        self._heading_integral = 0.0
        self._lateral_integral = 0.0
        self._last_heading_error = 0.0
        self._last_control_time = None
        
        # Tuning parameters
        self.heading_kp = 5.0
        self.heading_ki = 1.5
        self.heading_kd = 1.2
        self.lateral_kp = 0.4
        self.lateral_ki = 0.02
        self.max_correction = 250
        self.max_heading_correction = 45 # degrees
        
        # Threading
        self._running = False
        self._control_thread = None
        self._telemetry = deque(maxlen=2000)
        self._control_rate = 50 # Hz
        
        print("TrackController ready.")
    
    def _read_gyro(self) -> float:
        """Read gyro Z with bias correction and inversion."""
        _, _, gz = self._mpu.readGyroscopeMaster()
        gz = -(gz - self._gyro_bias)  # Inverted for this setup
        return gz if abs(gz) > 0.5 else 0.0
    
    def _update_position(self, dt: float):
        """Update position estimate from encoders + heading."""
        left_ticks = self._driver.left.ticks
        right_ticks = self._driver.right.ticks
        
        dl = (left_ticks - self._last_left_ticks) * MM_PER_TICK
        dr = -(right_ticks - self._last_right_ticks) * MM_PER_TICK # Right inverted
        
        self._last_left_ticks = left_ticks
        self._last_right_ticks = right_ticks
        
        dist = (dl + dr) / 2.0
        heading_rad = np.radians(self._heading)
        self._x += dist * np.cos(heading_rad)
        self._y += dist * np.sin(heading_rad)
    
    def _compute_correction(self, dt: float) -> float:
        """Compute wheel differential for track following."""
        if not self._correction_enabled:
            return 0.0
        
        # Lateral error -> target heading
        self._lateral_integral += self._y * dt
        self._lateral_integral = np.clip(self._lateral_integral, -500, 500)
        
        target_heading = -(self.lateral_kp * self._y + self.lateral_ki * self._lateral_integral)
        target_heading = np.clip(target_heading, -self.max_heading_correction, self.max_heading_correction)
        
        # Heading PID
        heading_error = self._heading - target_heading
        
        self._heading_integral += heading_error * dt
        self._heading_integral = np.clip(self._heading_integral, -100, 100)
        
        d_term = (heading_error - self._last_heading_error) / dt if dt > 0 else 0
        self._last_heading_error = heading_error
        
        correction = (self.heading_kp * heading_error +
                      self.heading_ki * self._heading_integral +
                      self.heading_kd * d_term)
        
        return np.clip(correction, -self.max_correction, self.max_correction)
    
    def _compute_turn_speeds(self) -> tuple[float, float]:
        """Compute wheel speeds for current turn."""
        if self._turn_radius == 0:
            # Pivot turn: wheels opposite directions
            # Flipped: positive direction = right turn = left forward, right backward
            left = self._turn_speed * self._turn_direction
            right = -self._turn_speed * self._turn_direction
        elif self._turn_radius == float('inf'):
            # Swing turn: one wheel stopped
            if self._turn_direction > 0: # Right turn
                left = self._turn_speed
                right = 0
            else: # Left turn
                left = 0
                right = self._turn_speed
        else:
            # Arc turn: both wheels, outer faster
            r = self._turn_radius
            w = WHEEL_BASE_MM / 2
            
            if self._turn_direction > 0:  # Right turn (turning around right side)
                left = self._turn_speed * (r + w) / r # Outer wheel
                right = self._turn_speed * (r - w) / r # Inner wheel
            else:  # Left turn
                left = self._turn_speed * (r - w) / r # Inner wheel
                right = self._turn_speed * (r + w) / r # Outer wheel
        
        return left, right
    
    def _control_loop(self):
        """Main control loop running at fixed rate."""
        period = 1.0 / self._control_rate
        last_time = time.monotonic()
        
        while self._running:
            now = time.monotonic()
            dt = now - last_time
            last_time = now
            
            # Acceleration ramp
            if self._current_speed < self._target_speed:
                self._current_speed = min(self._current_speed + self._accel_rate * dt, self._target_speed)
            elif self._current_speed > self._target_speed:
                self._current_speed = max(self._current_speed - self._accel_rate * dt, self._target_speed)
            
            # Update sensors
            gz = self._read_gyro()
            self._heading += gz * dt
            self._update_position(dt)
            
            # Compute and apply correction
            if self._turning:
                # Safety timeout: 10 seconds max for any turn
                if now - self._turn_start_time > 10.0:
                    print("\n[SAFETY] Turn timeout!")
                    self._turning = False
                    self._driver.stop()
                    continue
                
                # Check if turn complete (check distance to target, not direction)
                turned = self._heading - (self._turn_target - (self._turn_target - self._heading))
                remaining = self._turn_target - self._heading
                
                # Done when we've crossed the target
                if self._turn_direction > 0: # Right turn (heading increasing)
                    done = self._heading >= self._turn_target
                else:  # Left turn (heading decreasing)
                    done = self._heading <= self._turn_target
                
                if self._turn_direction > 0:  # Right turn
                    done = self._heading >= self._turn_target
                else:  # Left turn
                    done = self._heading <= self._turn_target
                
                if done:
                    self._turning = False
                    self._turn_complete = True
                    self._driver.stop()
                else:
                    # Slow down as we approach target
                    remaining = abs(self._turn_target - self._heading)
                    speed_factor = min(1.0, remaining / 15) # Slow down in last 15 degrees
                    speed_factor = max(0.5, speed_factor) # Minimum 50% speed
                    
                    left, right = self._compute_turn_speeds()
                    left *= speed_factor
                    right *= speed_factor
                    
                    # For pivot turns, use the driver directly with signed PWM
                    if self._turn_radius == 0:
                        # Pivot: directly control PWM direction
                        left_pwm = (abs(left) / 1000) * (1 if left >= 0 else -1)
                        right_pwm = (abs(right) / 1000) * (1 if right >= 0 else -1)
                        left_pwm = np.clip(left_pwm, -0.8, 0.8)
                        right_pwm = np.clip(right_pwm, -0.8, 0.8)
                        self._driver.left.write_pwm(left_pwm)
                        self._driver.right.write_pwm(right_pwm)
                    elif self._turn_radius == float('inf'):
                        # Swing turn: one wheel stopped, one moving – use direct PWM
                        left_pwm = (abs(left) / 800) if left > 0 else 0
                        right_pwm = (abs(right) / 800) if right > 0 else 0
                        left_pwm = min(0.9, left_pwm)
                        right_pwm = min(0.9, right_pwm)
                        self._driver.left.write_pwm(left_pwm)
                        self._driver.right.write_pwm(right_pwm)
                    else:
                        # Arc: both wheels forward, different speeds
                        self._driver.set_speed_tps(max(0, left), max(0, right))
            
            elif self._current_speed > 0:
                correction = self._compute_correction(dt)
                left_target = max(0, self._current_speed - correction)
                right_target = max(0, self._current_speed + correction)
                self._driver.set_speed_tps(left_target, right_target)
            
            # Log telemetry
            self._telemetry.append(RobotState(
                timestamp=now,
                x_mm=self._x, y_mm=self._y,
                heading_deg=self._heading,
                speed_left=self._driver.left.speed_tps,
                speed_right=self._driver.right.speed_tps
            ))
            
            # Maintain loop rate
            elapsed = time.monotonic() - now
            if elapsed < period:
                time.sleep(period - elapsed)
    
    # Public Interface
    def start(self):
        """Start the controller. Call this before any movement commands."""
        self._driver.start()
        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
    
    def drive(self, speed_tps: float, reset_position: bool = True):
        """
        Drive straight at the given speed.
        
        Args:
            speed_tps: Target speed in ticks per second
            reset_position: If True, reset position/heading tracking
        """
        if reset_position:
            self.reset_tracking()
        self._target_speed = abs(speed_tps)
        self._current_speed = 200 # Start above stiction zone
    
    def stop(self):
        """Stop driving."""
        self._target_speed = 0.0
        self._current_speed = 0.0
        self._turning = False
        self._driver.left.write_pwm(0) # Direct PWM stop
        self._driver.right.write_pwm(0)
        self._driver.stop()
    
    def pivot_turn(self, degrees: float, speed_tps: float = 300) -> None:
        """
        Spin in place.
        
        Args:
            degrees: Angle to turn (positive = right, negative = left)
            speed_tps: Wheel speed for turning (default 300 - conservative)
        """
        self._turn_target = self._heading + degrees
        self._turn_speed = speed_tps
        self._turn_radius = 0
        self._turn_direction = 1 if degrees > 0 else -1
        self._turn_complete = False
        self._turn_start_time = time.monotonic()
        self._turning = True
    
    def swing_turn(self, degrees: float, speed_tps: float = 300) -> None:
        """
        Turn with one wheel stopped (pivot around that wheel).
        
        Args:
            degrees: Angle to turn (positive = right, negative = left)
            speed_tps: Moving wheel speed (default 300 - conservative)
        """
        self._turn_target = self._heading + degrees
        self._turn_speed = speed_tps
        self._turn_radius = float('inf') # Signal for swing turn
        self._turn_direction = 1 if degrees > 0 else -1
        self._turn_complete = False
        self._turn_start_time = time.monotonic()
        self._turning = True
    
    def arc_turn(self, degrees: float, radius_mm: float, speed_tps: float = 400) -> None:
        """
        Turn in an arc of specified radius.
        
        Args:
            degrees: Angle to turn (positive = right, negative = left)
            radius_mm: Turn radius from robot center (must be > wheelbase/2)
            speed_tps: Base speed (outer wheel will be faster)
        """
        min_radius = WHEEL_BASE_MM / 2 + 10
        if radius_mm < min_radius:
            print(f"Warning: radius must be >= {min_radius}mm, using minimum")
            radius_mm = min_radius
        
        self._turn_target = self._heading + degrees
        self._turn_speed = speed_tps
        self._turn_radius = radius_mm
        self._turn_direction = 1 if degrees > 0 else -1
        self._turn_complete = False
        self._turn_start_time = time.monotonic()
        self._turning = True
    
    def turn(self, degrees: float, radius_mm: float = 0, speed_tps: float = 300) -> None:
        """
        Universal turn command.
        
        Args:
            degrees: Angle to turn (positive = right, negative = left)
            radius_mm: Turn radius. 0 = pivot, None/'swing' = swing turn, >0 = arc
            speed_tps: Base speed for turn
        """
        if radius_mm == 0:
            self.pivot_turn(degrees, speed_tps)
        elif radius_mm == float('inf') or radius_mm is None:
            self.swing_turn(degrees, speed_tps)
        else:
            self.arc_turn(degrees, radius_mm, speed_tps)
    
    def wait_for_turn(self, timeout: float = 10.0) -> bool:
        """
        Block until turn completes.
        
        Returns:
            True if turn completed, False if timeout
        """
        start = time.monotonic()
        while self._turning and (time.monotonic() - start) < timeout:
            time.sleep(0.02)
        return self._turn_complete
    
    @property
    def is_turning(self) -> bool:
        """Check if currently turning."""
        return self._turning
    
    # High-Level Navigation
    def drive_distance(self, distance_mm: float, speed_tps: float = 600) -> bool:
        """
        Drive straight for a specific distance, then stop.
        
        Args:
            distance_mm: Distance to travel (positive = forward)
            speed_tps: Driving speed
            
        Returns:
            True if completed, False if interrupted
        """
        self.reset_tracking()
        target_dist = abs(distance_mm)
        
        # Start driving using the normal drive method
        self.drive(speed_tps, reset_position=False)
        
        while self._running:
            current_dist = abs(self._x)
            
            if current_dist >= target_dist:
                self.stop()
                return True
            
            # Slow down in last 50mm
            remaining = target_dist - current_dist
            if remaining < 50:
                speed_factor = max(0.4, remaining / 50)
                self._target_speed = speed_tps * speed_factor
            
            time.sleep(0.02)
        
        return False
    
    def drive_to(self, x_mm: float, y_mm: float, speed_tps: float = 600) -> bool:
        """
        Drive to a specific (x, y) coordinate.
        
        Args:
            x_mm: Target X position
            y_mm: Target Y position
            speed_tps: Driving speed
            
        Returns:
            True if reached, False if interrupted
        """
        # Calculate angle to target
        dx = x_mm - self._x
        dy = y_mm - self._y
        target_heading = np.degrees(np.arctan2(dy, dx))
        distance = np.sqrt(dx**2 + dy**2)
        
        # Turn to face target
        angle_diff = target_heading - self._heading
        # Normalize to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        if abs(angle_diff) > 2: # Only turn if needed
            self.pivot_turn(angle_diff)
            if not self.wait_for_turn():
                return False
            time.sleep(0.2)
        
        # Drive to target
        return self.drive_distance(distance, speed_tps)
    
    def drive_square(self, size_mm: float, speed_tps: float = 600, turn_type: str = 'pivot') -> bool:
        """
        Drive in a square pattern.
        
        Args:
            size_mm: Side length of square
            speed_tps: Driving speed
            turn_type: 'pivot', 'swing', or 'arc'
            
        Returns:
            True if completed, False if interrupted
        """
        for i in range(4):
            # Drive one side
            if not self.drive_distance(size_mm, speed_tps):
                return False
            time.sleep(0.2)
            
            # Turn 90° right
            if turn_type == 'pivot':
                self.pivot_turn(90)
            elif turn_type == 'swing':
                self.swing_turn(90)
            elif turn_type == 'arc':
                self.arc_turn(90, radius_mm=size_mm / 4)
            else:
                self.pivot_turn(90)
            
            if not self.wait_for_turn():
                return False
            time.sleep(0.2)
        
        return True
    
    def drive_polygon(self, sides: int, size_mm: float, speed_tps: float = 600) -> bool:
        """
        Drive in a regular polygon pattern.
        
        Args:
            sides: Number of sides (3 = triangle, 6 = hexagon, etc.)
            size_mm: Side length
            speed_tps: Driving speed
            
        Returns:
            True if completed, False if interrupted
        """
        turn_angle = 360 / sides
        
        for i in range(sides):
            if not self.drive_distance(size_mm, speed_tps):
                return False
            time.sleep(0.2)
            
            self.pivot_turn(turn_angle)
            if not self.wait_for_turn():
                return False
            time.sleep(0.2)
        
        return True
    
    def drive_path(self, waypoints: list[tuple[float, float]], speed_tps: float = 600) -> bool:
        """
        Drive through a series of waypoints.
        
        Args:
            waypoints: List of (x_mm, y_mm) coordinates
            speed_tps: Driving speed
            
        Returns:
            True if completed, False if interrupted
        """
        for x, y in waypoints:
            if not self.drive_to(x, y, speed_tps):
                return False
            time.sleep(0.1)
        
        return True
    
    def shutdown(self):
        """Shutdown the controller. Call when done."""
        self._running = False # Stop control loop FIRST
        self._turning = False
        self._target_speed = 0.0
        self._current_speed = 0.0
        self._driver.left.write_pwm(0) # Direct PWM stop
        self._driver.right.write_pwm(0)
        self._driver.stop()
        if self._control_thread:
            self._control_thread.join(timeout=0.5)
        self._driver.disable()
    
    def reset_tracking(self):
        """Reset position and heading to zero."""
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._heading_integral = 0.0
        self._lateral_integral = 0.0
        self._last_heading_error = 0.0
        self._last_left_ticks = self._driver.left.ticks
        self._last_right_ticks = self._driver.right.ticks
    
    def enable_correction(self, enabled: bool = True):
        """Enable or disable track correction."""
        self._correction_enabled = enabled
        if enabled:
            self._heading_integral = 0.0
            self._lateral_integral = 0.0
    
    @property
    def state(self) -> RobotState:
        """Get current robot state."""
        if self._telemetry:
            return self._telemetry[-1]
        return RobotState(time.monotonic(), 0, 0, 0, 0, 0)
    
    @property
    def position(self) -> tuple[float, float]:
        """Get (x, y) position in mm."""
        return (self._x, self._y)
    
    @property
    def heading(self) -> float:
        """Get heading in degrees."""
        return self._heading
    
    @property
    def telemetry(self) -> list[RobotState]:
        """Get telemetry history."""
        return list(self._telemetry)

