#!/usr/bin/env python3
"""
Drive straight until an obstacle is detected, then stop.
"""
import time
import argparse
import numpy as np

from vision.depth import DepthEstimator
from old_tests.track_controller_old import TrackController


class ObstacleAvoider:
    """Drive until obstacle detected."""
    
    def __init__(
        self,
        stop_threshold: float = 0.15,
        slow_threshold: float = 0.25,
        cruise_speed: float = 500,
        slow_speed: float = 250,
        center_width: float = 0.4,
        use_percentile: bool = True,
        percentile: float = 10.0
    ):
        """
        Args:
            stop_threshold: Normalized depth below which to stop (0=close, 1=far)
            slow_threshold: Normalized depth below which to slow down
            cruise_speed: Normal driving speed (tps)
            slow_speed: Reduced speed when obstacle approaching
            center_width: Fraction of frame width to use for center detection
            use_percentile: Use percentile instead of mean (better for catching close objects)
            percentile: Which percentile to use (lower = more sensitive to close objects)
        """
        self.stop_threshold = stop_threshold
        self.slow_threshold = slow_threshold
        self.cruise_speed = cruise_speed
        self.slow_speed = slow_speed
        self.center_width = center_width
        self.use_percentile = use_percentile
        self.percentile = percentile
        
        print("Initializing obstacle avoider...")
        print(f"  Stop threshold:  {stop_threshold:.2f}")
        print(f"  Slow threshold:  {slow_threshold:.2f}")
        print(f"  Cruise speed:    {cruise_speed} tps")
        print(f"  Slow speed:      {slow_speed} tps")
        
        self.depth = DepthEstimator()
        self.controller = TrackController()
    
    def get_center_distance(self, depth_normalized: np.ndarray) -> float:
        """Get distance reading from center of frame."""
        h, w = depth_normalized.shape
        
        # Center region
        margin = int(w * (1 - self.center_width) / 2)
        center = depth_normalized[:, margin:w-margin]
        
        if self.use_percentile:
            # Use low percentile to catch the closest objects
            return float(np.percentile(center, self.percentile))
        else:
            return float(np.mean(center))
    
    def run(self, timeout: float = 30.0):
        """
        Drive forward until obstacle detected.
        
        Args:
            timeout: Maximum run time in seconds
        """
        print("\nStarting systems...")
        self.depth.start()
        self.controller.start()
        
        # Let camera/model warm up and show initial readings
        time.sleep(0.5)
        print("Warming up depth sensor...")
        for i in range(10):
            frame = self.depth.capture()
            d = self.get_center_distance(frame.normalized)
            print(f"  Warmup {i+1}/10: distance = {d:.3f}")
            time.sleep(0.1)
        
        # Check if path is clear before starting
        frame = self.depth.capture()
        initial_distance = self.get_center_distance(frame.normalized)
        print(f"\nInitial distance: {initial_distance:.3f}")
        
        if initial_distance < self.stop_threshold:
            print(f"WARNING: Path not clear! (reading {initial_distance:.3f} < threshold {self.stop_threshold})")
            resp = input("Start anyway? [y/N]: ").strip().lower()
            if resp != 'y':
                print("Aborted.")
                self.depth.stop()
                self.controller.shutdown()
                return
        
        print("\n" + "="*50)
        print("DRIVING - Press Ctrl+C to stop")
        print("="*50 + "\n")
        
        start_time = time.time()
        stopped = False
        
        try:
            # Start driving
            self.controller.drive(self.cruise_speed)
            
            while (time.time() - start_time) < timeout:
                # Get depth reading
                frame = self.depth.capture()
                distance = self.get_center_distance(frame.normalized)
                
                # Get position for logging
                x, y = self.controller.position
                heading = self.controller.heading
                
                # Decision logic
                if distance < self.stop_threshold:
                    # STOP - obstacle too close
                    self.controller.stop()
                    stopped = True
                    print(f"\n\n{'='*50}")
                    print(f"STOPPED - Obstacle detected!")
                    print(f"  Distance reading: {distance:.3f}")
                    print(f"  Position: ({x:.0f}, {y:.0f}) mm")
                    print(f"  Traveled: {x:.0f} mm")
                    print(f"{'='*50}")
                    break
                    
                elif distance < self.slow_threshold:
                    # SLOW - obstacle approaching
                    self.controller._target_speed = self.slow_speed
                    status = "SLOW"
                    
                else:
                    # CRUISE - path clear
                    self.controller._target_speed = self.cruise_speed
                    status = "CRUISE"
                
                # Status output
                bar_width = 30
                bar_fill = int((1 - distance) * bar_width)
                bar = "█" * bar_fill + "░" * (bar_width - bar_fill)
                
                print(f"\r[{bar}] d={distance:.3f} | {status:6s} | x={x:5.0f}mm | hdg={heading:+.1f}°  ", end="", flush=True)
                
                time.sleep(0.02)  # 50Hz loop
            
            if not stopped:
                self.controller.stop()
                print(f"\n\nTimeout reached ({timeout}s)")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            
        finally:
            self.controller.stop()
            self.controller.shutdown()
            self.depth.stop()
            print("Done.")
    
    def calibrate(self, distances_mm: list[int] = [150, 300, 500, 750, 1000]):
        """
        Interactive calibration routine.
        Place rover at known distances from a wall and record readings.
        """
        print("\nCalibration mode")
        print("Place the rover facing a flat wall.\n")
        
        self.depth.start()
        time.sleep(0.5)
        
        readings = []
        
        try:
            for dist in distances_mm:
                input(f"Place rover {dist}mm from wall, press Enter...")
                
                # Take multiple readings and average
                samples = []
                for _ in range(20):
                    frame = self.depth.capture()
                    samples.append(self.get_center_distance(frame.normalized))
                    time.sleep(0.05)
                
                avg = np.mean(samples)
                std = np.std(samples)
                readings.append((dist, avg, std))
                print(f"  {dist}mm -> {avg:.4f} (±{std:.4f})")
            
            print("\n" + "="*50)
            print("CALIBRATION RESULTS")
            print("="*50)
            print("\nDistance (mm) | Normalized | Std Dev")
            print("-" * 40)
            for dist, avg, std in readings:
                print(f"  {dist:5d}       |   {avg:.4f}   |  {std:.4f}")
            
            # Suggest thresholds
            if len(readings) >= 2:
                # Find reading closest to 300mm for stop threshold
                stop_reading = min(readings, key=lambda x: abs(x[0] - 300))[1]
                slow_reading = min(readings, key=lambda x: abs(x[0] - 600))[1]
                
                print(f"\nSuggested thresholds:")
                print(f"  stop_threshold = {stop_reading:.2f}  (based on ~300mm)")
                print(f"  slow_threshold = {slow_reading:.2f}  (based on ~600mm)")
                
        finally:
            self.depth.stop()


def main():
    parser = argparse.ArgumentParser(description="Drive until obstacle")
    parser.add_argument('--calibrate', action='store_true', help='Run calibration routine')
    parser.add_argument('--stop', type=float, default=0.15, help='Stop threshold (0-1)')
    parser.add_argument('--slow', type=float, default=0.25, help='Slow threshold (0-1)')
    parser.add_argument('--speed', type=float, default=500, help='Cruise speed (tps)')
    parser.add_argument('--timeout', type=float, default=30, help='Max runtime (seconds)')
    args = parser.parse_args()
    
    avoider = ObstacleAvoider(
        stop_threshold=args.stop,
        slow_threshold=args.slow,
        cruise_speed=args.speed
    )
    
    if args.calibrate:
        avoider.calibrate()
    else:
        avoider.run(timeout=args.timeout)


if __name__ == '__main__':
    main()