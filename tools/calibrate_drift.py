#!/usr/bin/env python3
"""
calibrate_drift.py — Measures natural motor drift at different speeds.

For each test speed, drives straight for 2 seconds with correction DISABLED,
then records the average PWM differential needed to keep both wheels matched.
Results are saved to config.yaml under 'drift_correction'.

Run this on a clear straight surface before using autonomous.py.
Usage: python tools/calibrate_drift.py
"""
import sys
import os
import time
import threading
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hardware.motor_controller import MotorDriver

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / 'config.yaml'

# Speeds to test (tps) — covers the range you actually use
TEST_SPEEDS = [400, 500, 600, 700]
SETTLE_TIME     = 0.5  # seconds to let speed stabilize before recording
MIN_SPEED_FLOOR = 200  # tps — never command below this
MIN_SAMPLES     = 10   # minimum samples before allowing a stop


def measure_drift_at_speed(driver: MotorDriver, speed_tps: float) -> float:
    """
    Drive at speed with no correction. A background thread watches for
    Enter key to stop — so Ctrl+C doesn't kill the whole script.
    """
    driver.set_speed_tps(speed_tps, speed_tps)
    time.sleep(SETTLE_TIME)

    samples = []
    stop_flag = threading.Event()

    def wait_for_enter():
        input()  # blocks until Enter
        stop_flag.set()

    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()

    print("  Rover moving — press Enter when you need to stop...")
    while not stop_flag.is_set():
        l =  driver.left.speed_tps
        r = -driver.right.speed_tps
        samples.append(l - r)
        time.sleep(0.05)

    driver.stop()
    time.sleep(0.5)

    if len(samples) < MIN_SAMPLES:
        print(f"  Too few samples ({len(samples)}) — skipping this speed")
        return None

    avg_diff = float(np.mean(samples))
    std_diff = float(np.std(samples))
    print(f"  {speed_tps:4d} tps | drift: {avg_diff:+7.2f} tps avg | std: {std_diff:.2f} | n={len(samples)}")
    return avg_diff


def main():
    print("=" * 50)
    print("DRIFT CALIBRATION")
    print("=" * 50)
    print(f"\nSpeeds to test: {TEST_SPEEDS}")
    print("For each speed: rover drives, you press Ctrl+C to stop it when")
    print("you're about to run out of space. Then reposition and continue.")
    print("\nPlace rover on a clear straight surface.")
    input("Press Enter to begin...\n")

    driver = MotorDriver()
    driver.start()

    results = {}

    try:
        for speed in TEST_SPEEDS:
            if speed < MIN_SPEED_FLOOR:
                print(f"Skipping {speed} tps (below stiction floor)")
                continue

            print(f"\n--- {speed} tps ---")
            input("Reposition rover to start, then press Enter...")
            print(f"Rover moving — press Ctrl+C when you need to stop...")
            drift = measure_drift_at_speed(driver, speed)

            if drift is not None:
                results[speed] = drift

    except KeyboardInterrupt:
        print("\n\nCalibration aborted early — saving whatever we have.")
    finally:
        driver.disable()

    if not results:
        print("No data collected.")
        return

    # Summarize
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for spd, drift in results.items():
        direction = "left faster (drifts right)" if drift > 0 else "right faster (drifts left)"
        print(f"  {spd} tps: {drift:+.2f} tps correction needed ({direction})")

    # Save to config.yaml
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['drift_correction'] = {
        int(k): round(v, 3) for k, v in results.items()
    }

    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\nSaved to {CONFIG_PATH} under 'drift_correction'.")
    print("Run autonomous.py — navigation will apply these at startup.")


if __name__ == '__main__':
    main()