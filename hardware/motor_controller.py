# """
# Hybrid motor control: Feedforward + light feedback correction.
# This gives smooth operation with precise encoder-based speed matching.
# """
# import yaml
# from pathlib import Path
# import time
# import threading
# from gpiozero import PWMOutputDevice, DigitalOutputDevice, RotaryEncoder
# from gpiozero.pins.lgpio import LGPIOFactory

# factory = LGPIOFactory()

# current_folder = Path(__file__).parent
# root_folder = current_folder.parent
# config_path = root_folder / 'config.yaml'
# with open(config_path, 'r') as file:
#     config = yaml.safe_load(file)

# # Motor specs
# ENCODER_CPR = 64
# GEAR_RATIO = 70
# TICKS_PER_WHEEL_REV = ENCODER_CPR * GEAR_RATIO


# class HybridController:
#     """Feedforward + feedback controller for smooth, precise speed control."""
#     def __init__(self, kp: float = 0.00003, ki: float = 0.000005):
#         self.kp = kp
#         self.ki = ki
#         self.integral = 0.0
#         self.prev_time = None
        
#         # Feedforward calibration: PWM needed per tps
#         # From your test: 750 tps at 0.3 PWM = 0.0004 PWM per tps
#         self.ff_gain = 0.0004

#     def compute(self, setpoint: float, measured: float) -> float:
#         now = time.monotonic()
#         if self.prev_time is None:
#             dt = 0.05
#         else:
#             dt = now - self.prev_time
#         self.prev_time = now

#         # Feedforward: main control effort
#         feedforward = setpoint * self.ff_gain
        
#         # Feedback: small corrections only
#         error = setpoint - measured
        
#         # Only correct significant errors
#         if abs(error) > 30:
#             p_term = self.kp * error
            
#             self.integral += error * dt
#             self.integral = max(-50, min(50, self.integral))
#             i_term = self.ki * self.integral
#         else:
#             p_term = 0
#             i_term = 0
        
#         output = feedforward + p_term + i_term
#         return max(-1.0, min(1.0, output))

#     def reset(self):
#         self.integral = 0.0
#         self.prev_time = None


# class Motor:
#     def __init__(self, motor_name: str, direction: int = 1) -> None:
#         assert motor_name in ('left', 'right')
#         self.name = motor_name
#         self.direction = direction
        
#         mcfg = config['motors'][motor_name]
#         self.rpwm = PWMOutputDevice(mcfg['rpwm'], pin_factory=factory, frequency=10000)
#         self.lpwm = PWMOutputDevice(mcfg['lpwm'], pin_factory=factory, frequency=10000)
#         self.en = DigitalOutputDevice(mcfg['en'], pin_factory=factory)
#         self.en.on()
        
#         ecfg = config['encoders'][motor_name]
#         self.encoder = RotaryEncoder(ecfg['a'], ecfg['b'], max_steps=0, pin_factory=factory)
        
#         self._pwm = 0.0
#         self._prev_ticks = 0
#         self._prev_time = time.monotonic()
#         self._speed_tps = 0.0
        
#         # Light smoothing
#         self._speed_history = []
#         self._history_size = 3

#     @property
#     def ticks(self) -> int:
#         return self.encoder.steps

#     @property
#     def speed_tps(self) -> float:
#         return self._speed_tps

#     def update_speed(self) -> float:
#         now = time.monotonic()
#         ticks = self.encoder.steps
#         dt = now - self._prev_time
        
#         if dt > 0:
#             raw_speed = (ticks - self._prev_ticks) / dt
            
#             self._speed_history.append(raw_speed)
#             if len(self._speed_history) > self._history_size:
#                 self._speed_history.pop(0)
#             self._speed_tps = sum(self._speed_history) / len(self._speed_history)
        
#         self._prev_ticks = ticks
#         self._prev_time = now
#         return self._speed_tps

#     def write_pwm(self, pwm: float) -> None:
#         pwm = pwm * self.direction # Apply direction multiplier
#         pwm = max(-1.0, min(1.0, pwm))
#         self._pwm = pwm
#         if pwm >= 0:
#             self.lpwm.value = 0
#             self.rpwm.value = pwm
#         else:
#             self.rpwm.value = 0
#             self.lpwm.value = -pwm

#     def stop(self) -> None:
#         self.write_pwm(0)

#     def disable(self) -> None:
#         self.stop()
#         self.en.off()

#     def reset_encoder(self) -> None:
#         self.encoder.steps = 0
#         self._prev_ticks = 0


# class MotorDriver:
#     def __init__(self) -> None:
#         self.left = Motor('left', direction=1) # Normal
#         self.right = Motor('right', direction=-1) # Reversed
        
#         # Hybrid controllers with light feedback
#         self.left_controller = HybridController(kp=0.00003, ki=0.000005)
#         self.right_controller = HybridController(kp=0.00003, ki=0.000005)
        
#         self._left_target = 0.0
#         self._right_target = 0.0
        
#         self._running = False
#         self._thread = None

#     def _control_loop(self):
#         while self._running:
#             left_actual = self.left.update_speed()
#             right_actual = self.right.update_speed()
            
#             left_pwm = self.left_controller.compute(self._left_target, left_actual)
#             right_pwm = self.right_controller.compute(self._right_target, right_actual)
            
#             self.left.write_pwm(left_pwm)
#             self.right.write_pwm(right_pwm)
            
#             time.sleep(0.05)

#     def start(self) -> None:
#         if not self._running:
#             self._running = True
#             self._thread = threading.Thread(target=self._control_loop, daemon=True)
#             self._thread.start()

#     def stop_loop(self) -> None:
#         self._running = False
#         if self._thread:
#             self._thread.join()

#     def set_speed_tps(self, left: float, right: float) -> None:
#         """Set target speeds in ticks per second."""
#         self._left_target = left
#         self._right_target = right

#     def forward(self, speed_tps: float = 700) -> None:
#         self.set_speed_tps(speed_tps, speed_tps)

#     def backward(self, speed_tps: float = 700) -> None:
#         self.set_speed_tps(-speed_tps, -speed_tps)

#     def turn_left(self, speed_tps: float = 500) -> None:
#         self.set_speed_tps(-speed_tps, speed_tps)

#     def turn_right(self, speed_tps: float = 500) -> None:
#         self.set_speed_tps(speed_tps, -speed_tps)

#     def stop(self) -> None:
#         self._left_target = 0.0
#         self._right_target = 0.0
#         self.left_controller.reset()
#         self.right_controller.reset()
#         self.left.stop()
#         self.right.stop()

#     def disable(self) -> None:
#         self.stop_loop()
#         self.left.disable()
#         self.right.disable()


# def test_straight_driving():
#     """Test encoder-controlled straight driving."""
#     driver = MotorDriver()
#     driver.start()
    
#     try:
#         print("\n=== Encoder-Controlled Straight Driving ===\n")
        
#         print("--- Test 1: Drive straight at 700 tps ---")
#         input("Press Enter to start...")
#         driver.forward(700)
        
#         print("\nMonitoring speed matching (should converge to ~0 difference):\n")
#         for i in range(50):
#             time.sleep(0.2)
#             diff = abs(driver.left.speed_tps - driver.right.speed_tps)
#             avg_speed = (driver.left.speed_tps + driver.right.speed_tps) / 2
#             print(f"[{i:2d}] L: {driver.left.speed_tps:6.1f} | R: {driver.right.speed_tps:6.1f} | "
#                   f"Diff: {diff:5.1f} | Avg: {avg_speed:6.1f} | Target: 700")
        
#         driver.stop()
#         time.sleep(2)
        
#         print("\n--- Test 2: Slower speed at 500 tps ---")
#         input("Press Enter to start...")
#         driver.forward(500)
        
#         for i in range(40):
#             time.sleep(0.2)
#             diff = abs(driver.left.speed_tps - driver.right.speed_tps)
#             avg_speed = (driver.left.speed_tps + driver.right.speed_tps) / 2
#             print(f"[{i:2d}] L: {driver.left.speed_tps:6.1f} | R: {driver.right.speed_tps:6.1f} | "
#                   f"Diff: {diff:5.1f} | Avg: {avg_speed:6.1f} | Target: 500")
        
#         driver.stop()
#         time.sleep(2)
        
#         print("\n--- Test 3: Higher speed at 900 tps ---")
#         input("Press Enter to start...")
#         driver.forward(900)
        
#         for i in range(40):
#             time.sleep(0.2)
#             diff = abs(driver.left.speed_tps - driver.right.speed_tps)
#             avg_speed = (driver.left.speed_tps + driver.right.speed_tps) / 2
#             print(f"[{i:2d}] L: {driver.left.speed_tps:6.1f} | R: {driver.right.speed_tps:6.1f} | "
#                   f"Diff: {diff:5.1f} | Avg: {avg_speed:6.1f} | Target: 900")
        
#         driver.stop()

#     except KeyboardInterrupt:
#         print("\nInterrupted.")
#     finally:
#         driver.disable()
#         print("Done.")

# if __name__ == '__main__':
#     test_straight_driving()
"""
Hybrid motor control: Feedforward + light feedback correction.
This gives smooth operation with precise encoder-based speed matching.
"""
import yaml
import signal
import atexit
from pathlib import Path
import time
import threading
from gpiozero import PWMOutputDevice, DigitalOutputDevice, RotaryEncoder
from gpiozero.pins.lgpio import LGPIOFactory

factory = LGPIOFactory()

current_folder = Path(__file__).parent
root_folder = current_folder.parent
config_path = root_folder / 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Motor specs
ENCODER_CPR = 64
GEAR_RATIO = 70
TICKS_PER_WHEEL_REV = ENCODER_CPR * GEAR_RATIO

# ---------------------------------------------------------------------------
# Global emergency stop registry
# Runs at the lowest level — before any high-level code gets a chance to act.
# Catches: Ctrl+C, SIGTERM (kill), SIGHUP (SSH disconnect), atexit (crash/normal exit)
# ---------------------------------------------------------------------------

_registered_motors: list = []  # All Motor instances register here

def _emergency_stop_all(signum=None, frame=None):
    """Kill all motors immediately. Called on any signal or process exit."""
    for m in _registered_motors:
        try:
            m.rpwm.value = 0
            m.lpwm.value = 0
            m.en.off()
        except Exception:
            pass  # Never raise during emergency stop
    if signum is not None:
        # Re-raise as default so process actually exits
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

# Register for every likely termination scenario
atexit.register(_emergency_stop_all)
signal.signal(signal.SIGINT,  _emergency_stop_all)  # Ctrl+C
signal.signal(signal.SIGTERM, _emergency_stop_all)  # kill / systemd stop
signal.signal(signal.SIGHUP,  _emergency_stop_all)  # SSH disconnect


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

class HybridController:
    """Feedforward + feedback controller for smooth, precise speed control."""
    def __init__(self, kp: float = 0.00003, ki: float = 0.000005):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.prev_time = None
        self.ff_gain = 0.0004

    def compute(self, setpoint: float, measured: float) -> float:
        now = time.monotonic()
        dt = 0.05 if self.prev_time is None else now - self.prev_time
        self.prev_time = now

        feedforward = setpoint * self.ff_gain
        error = setpoint - measured

        if abs(error) > 30:
            p_term = self.kp * error
            self.integral = max(-50, min(50, self.integral + error * dt))
            i_term = self.ki * self.integral
        else:
            p_term = 0
            i_term = 0

        return max(-1.0, min(1.0, feedforward + p_term + i_term))

    def reset(self):
        self.integral = 0.0
        self.prev_time = None


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------

class Motor:
    def __init__(self, motor_name: str, direction: int = 1) -> None:
        assert motor_name in ('left', 'right')
        self.name = motor_name
        self.direction = direction

        mcfg = config['motors'][motor_name]
        self.rpwm = PWMOutputDevice(mcfg['rpwm'], pin_factory=factory, frequency=10000)
        self.lpwm = PWMOutputDevice(mcfg['lpwm'], pin_factory=factory, frequency=10000)
        self.en   = DigitalOutputDevice(mcfg['en'], pin_factory=factory)
        self.en.on()

        ecfg = config['encoders'][motor_name]
        self.encoder = RotaryEncoder(ecfg['a'], ecfg['b'], max_steps=0, pin_factory=factory)

        self._pwm = 0.0
        self._prev_ticks = 0
        self._prev_time = time.monotonic()
        self._speed_tps = 0.0
        self._speed_history = []
        self._history_size = 3

        # Register with global emergency stop
        _registered_motors.append(self)

    @property
    def ticks(self) -> int:
        return self.encoder.steps

    @property
    def speed_tps(self) -> float:
        return self._speed_tps

    def update_speed(self) -> float:
        now = time.monotonic()
        ticks = self.encoder.steps
        dt = now - self._prev_time

        if dt > 0:
            raw_speed = (ticks - self._prev_ticks) / dt
            self._speed_history.append(raw_speed)
            if len(self._speed_history) > self._history_size:
                self._speed_history.pop(0)
            self._speed_tps = sum(self._speed_history) / len(self._speed_history)

        self._prev_ticks = ticks
        self._prev_time = now
        return self._speed_tps

    def write_pwm(self, pwm: float) -> None:
        pwm = max(-1.0, min(1.0, pwm * self.direction))
        self._pwm = pwm
        if pwm >= 0:
            self.lpwm.value = 0
            self.rpwm.value = pwm
        else:
            self.rpwm.value = 0
            self.lpwm.value = -pwm

    def stop(self) -> None:
        self.write_pwm(0)

    def disable(self) -> None:
        self.stop()
        self.en.off()

    def reset_encoder(self) -> None:
        self.encoder.steps = 0
        self._prev_ticks = 0


# ---------------------------------------------------------------------------
# MotorDriver (unchanged)
# ---------------------------------------------------------------------------

class MotorDriver:
    def __init__(self) -> None:
        self.left  = Motor('left',  direction=1)
        self.right = Motor('right', direction=-1)

        self.left_controller  = HybridController(kp=0.00003, ki=0.000005)
        self.right_controller = HybridController(kp=0.00003, ki=0.000005)

        self._left_target  = 0.0
        self._right_target = 0.0
        self._running = False
        self._thread  = None

    def _control_loop(self):
        while self._running:
            left_actual  = self.left.update_speed()
            right_actual = self.right.update_speed()

            left_pwm  = self.left_controller.compute(self._left_target,  left_actual)
            right_pwm = self.right_controller.compute(self._right_target, right_actual)

            self.left.write_pwm(left_pwm)
            self.right.write_pwm(right_pwm)

            time.sleep(0.05)

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._control_loop, daemon=True)
            self._thread.start()

    def stop_loop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join()

    def set_speed_tps(self, left: float, right: float) -> None:
        self._left_target  = left
        self._right_target = right

    def forward(self, speed_tps: float = 700) -> None:
        self.set_speed_tps(speed_tps, speed_tps)

    def backward(self, speed_tps: float = 700) -> None:
        self.set_speed_tps(-speed_tps, -speed_tps)

    def turn_left(self, speed_tps: float = 500) -> None:
        self.set_speed_tps(-speed_tps, speed_tps)

    def turn_right(self, speed_tps: float = 500) -> None:
        self.set_speed_tps(speed_tps, -speed_tps)

    def stop(self) -> None:
        self._left_target  = 0.0
        self._right_target = 0.0
        self.left_controller.reset()
        self.right_controller.reset()
        self.left.stop()
        self.right.stop()

    def disable(self) -> None:
        self.stop_loop()
        self.left.disable()
        self.right.disable()