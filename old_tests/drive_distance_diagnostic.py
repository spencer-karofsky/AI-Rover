#!/usr/bin/env python3
"""
Diagnostic: Is drive_distance failing due to X-axis drift bug?
Prints live X, Y, heading, and actual encoder distance so you can
see if X is lagging behind real travel.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from old_tests.track_controller_old import TrackController, MM_PER_TICK

TARGET_MM = 200
SPEED_TPS = 600

controller = TrackController()
controller.start()

print(f"\nTarget distance: {TARGET_MM}mm")
print(f"{'Time':>6} | {'X mm':>7} | {'Y mm':>7} | {'Hdg °':>7} | {'EncDist mm':>10} | {'Delta':>7}")
print("-" * 60)

# Snapshot encoder ticks at start
start_left  = controller._driver.left.ticks
start_right = controller._driver.right.ticks

controller.reset_tracking()
controller._target_speed = SPEED_TPS
controller._current_speed = 200

t0 = time.monotonic()
completed = False

try:
    while True:
        time.sleep(0.1)
        elapsed = time.monotonic() - t0

        # Real encoder-based distance (straight-line odometry, no heading assumption)
        dl = (controller._driver.left.ticks  - start_left)  * MM_PER_TICK
        dr = -(controller._driver.right.ticks - start_right) * MM_PER_TICK  # right inverted
        enc_dist = (dl + dr) / 2.0

        x   = controller.position[0]
        y   = controller.position[1]
        hdg = controller.heading

        # Gap between what drive_distance sees vs real travel
        delta = enc_dist - abs(x)

        print(f"{elapsed:6.1f} | {x:7.1f} | {y:7.1f} | {hdg:7.2f} | {enc_dist:10.1f} | {delta:7.1f}")

        # Replicate drive_distance's stop condition
        if abs(x) >= TARGET_MM:
            controller.stop()
            print("\n✓ Stopped via X condition")
            completed = True
            break

        # Safety: encoder says we've gone way past target
        if enc_dist > TARGET_MM * 1.5:
            controller.stop()
            print("\n✗ BUG CONFIRMED: Encoder passed 1.5x target but X never triggered stop")
            completed = True
            break

        # Safety timeout
        if elapsed > 15:
            controller.stop()
            print("\n✗ TIMEOUT: Never reached target distance")
            break

finally:
    controller.shutdown()

print(f"\n--- Summary ---")
print(f"Final X reported : {controller.position[0]:.1f}mm")
print(f"Final Y reported : {controller.position[1]:.1f}mm")
print(f"Final heading    : {controller.heading:.2f}°")
dl = (controller._driver.left.ticks  - start_left)  * MM_PER_TICK
dr = -(controller._driver.right.ticks - start_right) * MM_PER_TICK
enc_dist = (dl + dr) / 2.0
print(f"Encoder distance : {enc_dist:.1f}mm")
print(f"X vs enc delta   : {enc_dist - abs(controller.position[0]):.1f}mm")

if abs(controller.position[1]) > 20:
    print(f"\n⚠ Y drift of {controller.position[1]:.1f}mm — this is causing X to lag!")
if abs(controller.heading) > 5:
    print(f"⚠ Heading drift of {controller.heading:.2f}° — robot is not going straight")