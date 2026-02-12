import lgpio
import time

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_input(h, 23)
lgpio.gpio_claim_input(h, 24)

print("GPIO claimed successfully!")
print("Will run motors and check encoder...")

lgpio.gpiochip_close(h)