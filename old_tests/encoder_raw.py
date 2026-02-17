import lgpio
import time

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_input(h, 16)  # right encoder A
lgpio.gpio_claim_input(h, 26)  # right encoder B

# Right motor pins
lgpio.gpio_claim_output(h, 18)  # right rpwm  
lgpio.gpio_claim_output(h, 19)  # right lpwm
lgpio.gpio_claim_output(h, 6)   # right enable

print("Starting RIGHT motor...")

lgpio.gpio_write(h, 6, 1)   # enable on
lgpio.gpio_write(h, 19, 0)  # lpwm off

changes = 0
last16 = lgpio.gpio_read(h, 16)
last26 = lgpio.gpio_read(h, 26)

for i in range(100):
    lgpio.gpio_write(h, 18, 1)
    time.sleep(0.006)
    lgpio.gpio_write(h, 18, 0)
    time.sleep(0.014)
    
    v16 = lgpio.gpio_read(h, 16)
    v26 = lgpio.gpio_read(h, 26)
    if v16 != last16 or v26 != last26:
        changes += 1
        last16, last26 = v16, v26

lgpio.gpio_write(h, 18, 0)
lgpio.gpio_write(h, 6, 0)
lgpio.gpiochip_close(h)

print(f"Right encoder changes: {changes}")