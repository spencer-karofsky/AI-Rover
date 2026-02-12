#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/spencer/ai_rover')
from hardware.motor_controller import MotorDriver
import time

driver = MotorDriver()
driver.start()

print("Testing LEFT motor only...")
driver.set_speed_tps(700, 0)
time.sleep(2)
driver.stop()
time.sleep(1)

print("Testing RIGHT motor only...")
driver.set_speed_tps(0, 700)
time.sleep(2)
driver.stop()
time.sleep(1)

print("Testing BOTH motors...")
driver.set_speed_tps(700, 700)
time.sleep(2)
driver.stop()

driver.disable()
print("Done")
