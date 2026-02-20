#!/usr/bin/env python3
"""
autonomous.py — Fixed sequence: forward 0.5m, turn 90°, forward 1m.
All hardware management, safety, and threading handled by RoverCore.
"""
import time
from hardware.rover import RoverCore


def main():
    rover = RoverCore(enable_depth=True, depth_output='depth_recording.avi')
    rover.start(warmup=3.0)

    try:
        print("Step 1: Drive forward 500mm...")
        rover.drive_distance(200, speed_tps=500)
        time.sleep(0.4)

        print("Step 2: Pivot turn 90° right...")
        rover.swing_turn(30, speed_tps=280)
        rover.wait_for_turn(timeout=8.0)
        time.sleep(0.4)

        print("Step 3: Drive forward 1000mm...")
        rover.drive_distance(200, speed_tps=500)

        print("Sequence complete.")
        rover.drive_distance(1000, speed_tps=500)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Emergency stop!")
    finally:
        rover.shutdown()


if __name__ == '__main__':
    main()