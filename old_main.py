#!/usr/bin/env python3
"""
AI Rover - Hailo depth estimation + Motor control + IMU stabilization
"""
import cv2
import numpy as np
from picamera2 import Picamera2
import sys
import tty
import termios
import threading
import time
import signal
import os

from hailo_platform import VDevice, HailoSchedulingAlgorithm
from mpu9250_jmdev.registers import *
from mpu9250_jmdev.mpu_9250 import MPU9250

from hardware.motor_controller import MotorDriver

os.environ['DISPLAY'] = ':0'


class IMU:
    def __init__(self):
        self.mpu = MPU9250(
            address_ak=AK8963_ADDRESS,
            address_mpu_master=MPU9050_ADDRESS_68,
            address_mpu_slave=None,
            bus=1,
            gfs=GFS_1000,
            afs=AFS_8G,
            mfs=AK8963_BIT_16,
            mode=AK8963_MODE_C100HZ
        )
        self.mpu.configure()
        self.gyro_offset = [0, 0, 0]
        self.mag_offset = [0, 0, 0]
        self.mag_scale = [1, 1, 1]
        
    def calibrate(self, samples=100):
        print("Calibrating IMU - keep rover still...")
        gx_off, gy_off, gz_off = 0, 0, 0
        for _ in range(samples):
            gx, gy, gz = self.mpu.readGyroscopeMaster()
            gx_off += gx
            gy_off += gy
            gz_off += gz
            time.sleep(0.02)
        self.gyro_offset = [gx_off/samples, gy_off/samples, gz_off/samples]
        print(f"Gyro offsets: X={self.gyro_offset[0]:.2f} Y={self.gyro_offset[1]:.2f} Z={self.gyro_offset[2]:.2f}")
    
    def calibrate_mag(self, duration=10):
        print(f"Calibrating magnetometer - rotate rover slowly for {duration}s...")
        mag_min = [float('inf')] * 3
        mag_max = [float('-inf')] * 3
        start = time.time()
        
        while time.time() - start < duration:
            mx, my, mz = self.mpu.readMagnetometerMaster()
            mag_min[0] = min(mag_min[0], mx)
            mag_min[1] = min(mag_min[1], my)
            mag_min[2] = min(mag_min[2], mz)
            mag_max[0] = max(mag_max[0], mx)
            mag_max[1] = max(mag_max[1], my)
            mag_max[2] = max(mag_max[2], mz)
            remaining = duration - (time.time() - start)
            print(f"\rRotate rover... {remaining:.1f}s remaining", end="", flush=True)
            time.sleep(0.05)
        
        self.mag_offset = [(mag_min[i] + mag_max[i]) / 2 for i in range(3)]
        self.mag_scale = [(mag_max[i] - mag_min[i]) / 2 for i in range(3)]
        self.mag_scale = [s if s > 0 else 1 for s in self.mag_scale]
        print(f"\nMag offsets: {self.mag_offset}")
        print(f"Mag scale: {self.mag_scale}")
    
    def read_gyro(self):
        gx, gy, gz = self.mpu.readGyroscopeMaster()
        return (
            gx - self.gyro_offset[0],
            gy - self.gyro_offset[1],
            gz - self.gyro_offset[2]
        )
    
    def read_accel(self):
        return self.mpu.readAccelerometerMaster()
    
    def read_mag(self):
        mx, my, mz = self.mpu.readMagnetometerMaster()
        return (
            (mx - self.mag_offset[0]) / self.mag_scale[0],
            (my - self.mag_offset[1]) / self.mag_scale[1],
            (mz - self.mag_offset[2]) / self.mag_scale[2]
        )
    
    def get_heading(self):
        mx, my, mz = self.read_mag()
        heading = np.arctan2(my, mx) * 180 / np.pi
        heading = (heading + 360) % 360
        return heading


class StraightLineController:
    def __init__(self, kp=2.0, ki=0.1, kd=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0
        self.target_heading = 0
        self.current_heading = 0
        self.last_time = None
        
    def reset(self):
        self.integral = 0
        self.last_error = 0
        self.target_heading = 0
        self.current_heading = 0
        self.last_time = None
    
    def update(self, gyro_z):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            return 0
        
        dt = now - self.last_time
        self.last_time = now
        
        self.current_heading += gyro_z * dt
        error = self.current_heading - self.target_heading
        
        self.integral += error * dt
        self.integral = max(-50, min(50, self.integral))
        
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error
        
        correction = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-200, min(200, correction))


class HailoDepth:
    """Hailo-accelerated depth estimation using SCDepthV3"""
    def __init__(self, hef_path='/home/spencer/ai_rover/scdepthv3.hef'):
        print("Loading Hailo depth model...")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        self.vdevice = VDevice(params)
        self.infer_model = self.vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)
        self.configured_model = self.infer_model.configure()
        self.output_name = self.infer_model.output().name
        self.output_shape = self.infer_model.output().shape
        print(f"Hailo depth ready - input: {self.infer_model.input().shape}, output: {self.output_shape}")
    
    def infer(self, frame_rgb):
        """Run depth inference on RGB frame (320x256x3)
        Returns: depth map (256x320) as uint16, lower values = closer
        """
        output_buffer = np.empty(self.output_shape, dtype=np.uint16)
        bindings = self.configured_model.create_bindings(
            output_buffers={self.output_name: output_buffer}
        )
        bindings.input().set_buffer(frame_rgb)
        self.configured_model.run([bindings], timeout=1000)
        return output_buffer.squeeze()


class AIRover:
    def __init__(self):
        print("="*40)
        print("AI ROVER STARTING (Hailo Accelerated)")
        print("="*40)
        
        print("[1/4] Starting camera...")
        self.picam2 = Picamera2()
        # 320x256 to match Hailo input
        config = self.picam2.create_preview_configuration(
            main={"size": (320, 256), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        print("[2/4] Loading Hailo depth model...")
        self.depth_model = HailoDepth()
        
        print("[3/4] Starting motors...")
        self.driver = MotorDriver()
        self.driver.start()
        
        print("[4/4] Initializing IMU...")
        self.imu = IMU()
        self.imu.calibrate()
        self.heading_controller = StraightLineController(kp=3.0, ki=0.2, kd=0.8)
        self.compass_heading = 0
        self.mag_calibrated = False
        
        self.running = True
        self.recording = False
        self.video_writer = None
        self.latest_frame = None
        self.move_duration = 2.0
        
        # Movement state
        self.is_moving = False
        self.movement_thread = None
        
        # Frame: RGB(320) + Depth(320) + Colorbar(40) = 680 x 256
        self.frame_width = 680
        self.frame_height = 256
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("="*40)
        print("READY")
        print("="*40)
    
    def signal_handler(self, sig, frame):
        self.running = False
    
    def drive_straight(self, speed_tps, duration, reverse=False):
        self.is_moving = True
        
        if reverse:
            self.driver.backward(speed_tps)
        else:
            self.driver.forward(speed_tps)
        
        start = time.time()
        while self.is_moving and (time.time() - start) < duration:
            time.sleep(0.05)
        
        self.driver.stop()
        self.is_moving = False
    
    def turn_degrees(self, degrees, speed_tps=400):
        self.is_moving = True
        
        gz_off = 0
        for _ in range(20):
            _, _, gz = self.imu.mpu.readGyroscopeMaster()
            gz_off += gz
            time.sleep(0.02)
        gz_off /= 20
        
        turned = 0
        last_time = time.time()
        
        direction = 1 if degrees > 0 else -1
        target = abs(degrees)
        
        min_speed = 150
        decel_zone = 30
        
        while self.is_moving and turned < target:
            now = time.time()
            dt = now - last_time
            last_time = now
            
            _, _, gz = self.imu.mpu.readGyroscopeMaster()
            gz -= gz_off
            turned += abs(gz) * dt
            
            remaining = target - turned
            if remaining < decel_zone:
                speed = min_speed + (speed_tps - min_speed) * (remaining / decel_zone)
            else:
                speed = speed_tps
            
            speed = max(min_speed, speed)
            
            if direction > 0:
                self.driver.turn_right(speed)
            else:
                self.driver.turn_left(speed)
            
            time.sleep(0.02)
        
        self.driver.stop()
        self.is_moving = False
    
    def timed_move(self, action_type, **kwargs):
        if action_type == 'forward':
            self.drive_straight(kwargs.get('speed', 700), self.move_duration, reverse=False)
        elif action_type == 'backward':
            self.drive_straight(kwargs.get('speed', 700), self.move_duration, reverse=True)
        elif action_type == 'left':
            self.turn_degrees(-90, kwargs.get('speed', 300))
        elif action_type == 'right':
            self.turn_degrees(90, kwargs.get('speed', 300))
    
    def get_depth(self, frame_rgb):
        """Get depth from Hailo - returns uint16 depth map, lower = closer"""
        return self.depth_model.infer(frame_rgb)
    
    def colorize_depth(self, depth):
        """Colorize Hailo depth map (uint16, lower = closer)"""
        d = depth.astype(np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        d = (d * 255).astype(np.uint8)
        return cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)
    
    def draw_compass(self, frame, heading, cx, cy, radius=30):
        cv2.circle(frame, (cx, cy), radius, (40, 40, 40), -1)
        cv2.circle(frame, (cx, cy), radius, (100, 100, 100), 2)
        
        angle_rad = np.radians(-heading + 90)
        nx = int(cx + radius * 0.7 * np.cos(angle_rad))
        ny = int(cy - radius * 0.7 * np.sin(angle_rad))
        cv2.line(frame, (cx, cy), (nx, ny), (0, 0, 255), 2)
        
        sx = int(cx - radius * 0.5 * np.cos(angle_rad))
        sy = int(cy + radius * 0.5 * np.sin(angle_rad))
        cv2.line(frame, (cx, cy), (sx, sy), (200, 200, 200), 1)
        
        cv2.putText(frame, 'N', (cx-5, cy-radius-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(frame, f'{heading:.0f}', (cx-12, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def create_colorbar(self, height=256):
        colorbar = np.zeros((height, 40, 3), dtype=np.uint8)
        for i in range(height):
            val = int(255 * i / height)  # Flipped: top = close (dark), bottom = far (bright)
            colorbar[i, :15, :] = cv2.applyColorMap(
                np.array([[val]], dtype=np.uint8), cv2.COLORMAP_MAGMA)[0, 0]
        cv2.putText(colorbar, 'near', (16, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(colorbar, 'far', (16, height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return colorbar
    
    def keyboard_control(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self.running:
                ch = sys.stdin.read(1)
                if ch == 'w':
                    if self.movement_thread and self.movement_thread.is_alive():
                        self.is_moving = False
                        self.movement_thread.join(timeout=0.1)
                    self.movement_thread = threading.Thread(
                        target=self.timed_move, args=('forward',), daemon=True)
                    self.movement_thread.start()
                elif ch == 's':
                    if self.movement_thread and self.movement_thread.is_alive():
                        self.is_moving = False
                        self.movement_thread.join(timeout=0.1)
                    self.movement_thread = threading.Thread(
                        target=self.timed_move, args=('backward',), daemon=True)
                    self.movement_thread.start()
                elif ch == 'a':
                    if self.movement_thread and self.movement_thread.is_alive():
                        self.is_moving = False
                        self.movement_thread.join(timeout=0.1)
                    self.movement_thread = threading.Thread(
                        target=self.timed_move, args=('left',), daemon=True)
                    self.movement_thread.start()
                elif ch == 'd':
                    if self.movement_thread and self.movement_thread.is_alive():
                        self.is_moving = False
                        self.movement_thread.join(timeout=0.1)
                    self.movement_thread = threading.Thread(
                        target=self.timed_move, args=('right',), daemon=True)
                    self.movement_thread.start()
                elif ch == ' ':
                    self.is_moving = False
                    self.driver.stop()
                elif ch == 'c':
                    print("\n")
                    self.imu.calibrate_mag(duration=10)
                    self.mag_calibrated = True
                    print("Magnetometer calibrated!")
                elif ch == 'r':
                    self.toggle_recording()
                elif ch == 'q' or ch == '\x03':
                    self.running = False
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    
    def toggle_recording(self):
        if not self.recording:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                f'/home/spencer/ai_rover/recordings/depth_{timestamp}.mp4',
                fourcc, 30.0, (self.frame_width, self.frame_height)
            )
            self.recording = True
            print("\n[REC ON]")
        else:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("\n[REC OFF]")
    

    
    def run(self):
        print("\nW/A/S/D=Drive | SPACE=Stop | C=Calibrate Compass | R=Record | Q=Quit")
        print("Hailo depth: ACTIVE\n")
        
        threading.Thread(target=self.keyboard_control, daemon=True).start()
        
        cv2.namedWindow("AI Rover", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("AI Rover", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        colorbar = self.create_colorbar()
        frame_count = 0
        fps_time = time.time()
        fps = 0
        
        try:
            while self.running:
                # Capture frame (RGB format from picamera2)
                frame_rgb = cv2.flip(self.picam2.capture_array(), 0)
                
                start = time.time()
                
                # Get depth from Hailo
                depth = self.get_depth(frame_rgb)
                depth_color = self.colorize_depth(depth)
                
                elapsed = time.time() - start
                
                # IMU data
                gx, gy, gz = self.imu.read_gyro()
                if self.mag_calibrated:
                    self.compass_heading = self.imu.get_heading()
                
                # Annotate RGB frame
                cv2.putText(frame_rgb, 'RGB', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame_rgb, f'Yaw: {gz:.1f}/s', (10, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                if self.mag_calibrated:
                    self.draw_compass(frame_rgb, self.compass_heading, 280, 40, radius=30)
                else:
                    cv2.putText(frame_rgb, 'C=Cal', (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                
                # Annotate depth frame
                cv2.putText(depth_color, f'{elapsed*1000:.1f}ms', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Combine frames
                combined = np.hstack([frame_rgb, depth_color, colorbar])
                
                if self.recording:
                    cv2.circle(combined, (combined.shape[1]-20, 20), 10, (0, 0, 255), -1)
                    self.video_writer.write(combined)
                
                self.latest_frame = combined.copy()
                
                # Display on screen
                cv2.imshow("AI Rover", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                
                # FPS calculation
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - fps_time)
                    fps_time = time.time()
                
                rec = "REC" if self.recording else "   "
                hdg = f"Hdg:{self.compass_heading:3.0f}°" if self.mag_calibrated else "Hdg:---"
                print(f"\r[{rec}] {fps:.1f} FPS | {elapsed*1000:.1f}ms | {hdg} | Yaw: {gz:+.1f}°/s   ", end="", flush=True)
                
        finally:
            self.shutdown()
    
    def shutdown(self):
        print("\n\nShutting down...")
        self.running = False
        self.is_moving = False
        self.driver.stop()
        self.driver.disable()
        self.picam2.stop()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        try:
            os.system('stty sane')
        except:
            pass
        print("Done.")


if __name__ == '__main__':
    rover = AIRover()
    rover.run()