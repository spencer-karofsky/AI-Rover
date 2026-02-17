from old_tests.track_controller_old import TrackController

controller = TrackController()
controller.start()

print("Arc 90° right, 200mm radius...")
controller.arc_turn(90, radius_mm=200)
controller.wait_for_turn()
print(f"Heading: {controller.heading:.1f}°")

controller.shutdown()