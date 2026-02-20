import numpy as np
import cv2
from hailo_platform import HEF, VDevice, HailoStreamInterface

# Load the model you just compiled
hef = HEF('depth_anything_v2.hef')

with VDevice() as target:
    # Configure the Hailo-10H
    configure_params = target.create_configure_params(hef)
    network_group = target.configure(hef, configure_params)[0]
    
    # Setup input/output streams
    input_vstream_params = network_group.create_input_vstream_params()
    output_vstream_params = network_group.create_output_vstream_params()
    
    # Start the camera (adjust index if needed)
    cap = cv2.VideoCapture(0)

    with network_group.activate_context(), \
         target.create_input_vstream(input_vstream_params) as [input_vstream], \
         target.create_output_vstream(output_vstream_params) as [output_vstream]:
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Pre-process: Resize to 518x518
            resized_frame = cv2.resize(frame, (518, 518))
            input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0
            
            # RUN INFERENCE
            input_vstream.send(input_data)
            output_data = output_vstream.recv()
            
            # Post-process: Normalize for visualization
            depth_map = output_data[0].squeeze()
            depth_min, depth_max = depth_map.min(), depth_map.max()
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
            depth_viz = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGENTA)
            
            cv2.imshow('Robot Depth Vision', depth_color)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()