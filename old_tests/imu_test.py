from mpu9250_jmdev.registers import *
from mpu9250_jmdev.mpu_9250 import MPU9250
import time

mpu = MPU9250(
    address_ak=AK8963_ADDRESS,
    address_mpu_master=MPU9050_ADDRESS_68,
    address_mpu_slave=None,
    bus=1,
    gfs=GFS_1000,
    afs=AFS_8G,
    mfs=AK8963_BIT_16,
    mode=AK8963_MODE_C100HZ
)

mpu.configure()

# Calibrate gyro
print("Calibrating gyro - keep sensor still for 2 seconds...")
gx_off, gy_off, gz_off = 0, 0, 0
samples = 100
for _ in range(samples):
    gx, gy, gz = mpu.readGyroscopeMaster()
    gx_off += gx
    gy_off += gy
    gz_off += gz
    time.sleep(0.02)
gx_off /= samples
gy_off /= samples
gz_off /= samples
print(f"Gyro offsets: X={gx_off:.2f} Y={gy_off:.2f} Z={gz_off:.2f}\n")

print("MPU-9250 Calibrated - Press Ctrl+C to exit\n")

while True:
    ax, ay, az = mpu.readAccelerometerMaster()
    gx, gy, gz = mpu.readGyroscopeMaster()
    mx, my, mz = mpu.readMagnetometerMaster()
    
    # Apply gyro calibration
    gx -= gx_off
    gy -= gy_off
    gz -= gz_off
    
    print(f"Accel: X={ax:7.3f} Y={ay:7.3f} Z={az:7.3f} g")
    print(f"Gyro:  X={gx:7.2f} Y={gy:7.2f} Z={gz:7.2f} °/s")
    print(f"Mag:   X={mx:7.1f} Y={my:7.1f} Z={mz:7.1f} µT")
    print("-" * 45)
    
    time.sleep(0.5)