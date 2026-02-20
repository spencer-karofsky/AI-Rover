#!/usr/bin/env python3
"""
Diagnostic viewer for StereoNet depth fusion.
Controls:
  'l' - Toggle alignment lines
  'x' - Swap Left/Right inputs
  'q' - Quit
"""
import cv2
import numpy as np
from stereo import StereoCamera
from depth_fusion_new import DepthFusion


def run_diagnostic():
    stereo = StereoCamera(cam_left=1, cam_right=0)
    stereo.start()
    fusion = DepthFusion()

    print("--- Diagnostic Mode ---")
    print(" 'l' - Toggle Alignment Lines")
    print(" 'x' - Swap Left/Right Inputs")
    print(" 'q' - Quit")

    show_lines = True
    swap_inputs = False

    try:
        while True:
            left_raw, right_raw = stereo.capture_raw()
            left_rect, right_rect = stereo.rectify(left_raw, right_raw)

            # Swap test - keep track of what's actually going in
            if swap_inputs:
                in_l, in_r = right_rect, left_rect
            else:
                in_l, in_r = left_rect, right_rect

            frame_data = fusion.capture(in_l, in_r)

            # Pass in_r so the right panel reflects any swap
            vis = fusion.create_debug_view(frame_data, in_r, show_lines=show_lines)

            # Swap indicator
            if swap_inputs:
                cv2.putText(vis, "*** INPUTS SWAPPED ***", (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Diagnostic", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                show_lines = not show_lines
                print(f"Alignment lines: {'ON' if show_lines else 'OFF'}")
            elif key == ord('x'):
                swap_inputs = not swap_inputs
                print(f"Inputs swapped: {swap_inputs}")

    finally:
        stereo.stop()
        # DepthFusion has no stop() - just release the vdevice cleanly
        try:
            del fusion
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_diagnostic()