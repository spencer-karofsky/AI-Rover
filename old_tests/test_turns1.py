#!/usr/bin/env python3
"""Test navigation: 0.5m forward, 90° right, 1m forward."""

from old_tests.track_controller_old import TrackController
import time

controller = TrackController()
controller.start()

try:
    print("\n=== Navigation Test ===\n")
    
    # Drive 0.5m forward
    print("Driving 500mm forward...")
    controller.drive_distance(500)
    print(f"Position: ({controller.position[0]:.0f}, {controller.position[1]:.0f})mm")
    time.sleep(0.5)
    
    # Turn 90° right
    print("\nTurning 90° right...")
    controller.pivot_turn(90)
    controller.wait_for_turn()
    print(f"Heading: {controller.heading:.1f}°")
    time.sleep(0.5)
    
    # Drive 1m forward
    print("\nDriving 1000mm forward...")
    controller.drive_distance(1000)
    print(f"Position: ({controller.position[0]:.0f}, {controller.position[1]:.0f})mm")
    
    print("\n=== Complete ===")

except KeyboardInterrupt:
    print("\nInterrupted.")
finally:
    controller.stop()
    controller.shutdown()