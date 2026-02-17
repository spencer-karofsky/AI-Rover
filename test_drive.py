from brain.navigation import TrackController


if __name__ == '__main__':
    control = TrackController()
    control.start()
    control.drive_distance(200.0)
