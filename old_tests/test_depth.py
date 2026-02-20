import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface

# 1. Load the HEF you are currently compiling
hef = HEF('depth_anything.hef')

with VDevice() as target:
    # 2. Configure the NPU
    configure_params = target.create_configure_params(hef)
    network_group = target.configure(hef, configure_params)[0]
    
    # 3. Setup I/O streams
    input_vstream_params = network_group.make_input_vstream_params()
    output_vstream_params = network_group.make_output_vstream_params()
    
    # This is where we handle the 3 contexts automatically!
    with network_group.activate():
        print("Hailo-10H is active. Ready for frames.")
        # Your inference loop will go here