from vision.depth import DepthEstimator

depth = DepthEstimator()

# In your control loop:
result = depth.estimate(frame_rgb)
left, center, right = depth.get_obstacle_distances(result.normalized)

if center < 0.2:  # Obstacle ahead
    controller.stop()
    # Decide turn direction based on left vs right