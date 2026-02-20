import cv2
import time
import numpy as np
from stereo import StereoCamera
from depth_fusion_new import DepthFusion

def test_hailo_depth():
    print("Initializing hardware for test...")
    # 1. Start Stereo Cameras
    stereo = StereoCamera(cam_left=1, cam_right=0)
    stereo.start()

    # 2. Start new DepthFusion (StereoNet on Hailo-10H)
    fusion = DepthFusion()

    print("\n--- Hailo StereoNet Test ---")
    print("Press 'q' to quit. Press 's' to save a test frame.")
    
    try:
        while True:
            t_start = time.perf_counter()
            
            # 3. Capture & Rectify (From your existing stereo.py logic)
            left_raw, right_raw = stereo.capture_raw()
            left_rect, right_rect = stereo.rectify(left_raw, right_raw)
            
            # 4. Feed to the new DepthFusion
            # (Note: passing rectified frames is critical)
            frame_data = fusion.capture(left_rect, right_rect)
            
            # 5. Measure Performance
            total_ms = (time.perf_counter() - t_start) * 1000
            fps = 1000 / total_ms
            
            # 6. Obstacle check
            l, c, r = fusion.get_obstacle_distances_mm(frame_data.depth_metric)
            
            # 7. Visualization
            vis = fusion.create_debug_view(frame_data)
            
            # Overlay FPS and Distance
            cv2.putText(vis, f"Total FPS: {fps:.1f} | Hailo: {frame_data.hailo_ms:.1f}ms", 
                        (10, vis.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Hailo StereoNet Test", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("hailo_test_rgb.jpg", left_rect)
                cv2.imwrite("hailo_test_depth.jpg", frame_data.colorized)
                print("Test frames saved.")

    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        stereo.stop()
        fusion.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_hailo_depth()