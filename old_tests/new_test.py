from track_controller import TrackController
import time

controller = TrackController()
controller.start()

print("Driving 500mm...")
controller.drive_distance(500)
print(f"Final X: {controller.position[0]:.0f}mm")

controller.shutdown()