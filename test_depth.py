#!/usr/bin/env python3
"""
Side-by-side comparison: Depth-Anything-V2 (CPU) vs SCDepthV3 (Hailo)
Display on connected monitor via cv2.imshow
"""
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from picamera2 import Picamera2
from libcamera import Transform
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams
import time
import os

os.environ['DISPLAY'] = ':0'


def colorize_depth(depth, power=-0.5):
    """Normalize and colorize depth map"""
    d = depth.astype(np.float32)
    d = np.nan_to_num(d, nan=0, posinf=0, neginf=0)
    if d.max() - d.min() > 1e-6:
        pwr = np.power(d + 1e-6, power)
        norm = (pwr - pwr.min()) / (pwr.max() - pwr.min())
    else:
        norm = np.zeros_like(d)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main():
    print("=" * 50)
    print("DEPTH COMPARISON: CPU vs HAILO")
    print("=" * 50)
    
    # --- Camera setup (256x320 to match Hailo input) ---
    print("[1/4] Starting camera...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (320, 256), "format": "RGB888"},
        transform=Transform(hflip=True, vflip=True)
    )
    picam2.configure(config)
    picam2.start()
    
    # --- Hailo setup ---
    print("[2/4] Loading Hailo model (SCDepthV3)...")
    hef_path = '/home/spencer/ai_rover/scdepthv3.hef'
    hef = HEF(hef_path)
    input_vstream_infos = hef.get_input_vstream_infos()
    output_vstream_infos = hef.get_output_vstream_infos()
    print(f"    Hailo input: {input_vstream_infos[0].shape}")
    print(f"    Hailo output: {output_vstream_infos[0].shape}")
    
    # --- CPU model setup ---
    print("[3/4] Loading CPU model (Depth-Anything-V2)...")
    processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf",
        use_fast=True
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    model.eval()
    print("    CPU model loaded")
    
    print("[4/4] Opening display...")
    cv2.namedWindow("Depth Comparison", cv2.WINDOW_NORMAL)
    
    print("=" * 50)
    print("Press 'q' to quit")
    print("=" * 50)
    
    # Timing trackers
    hailo_times = []
    cpu_times = []
    frame_count = 0
    
    with VDevice() as device:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = device.configure(hef, configure_params)[0]
        
        with network_group.activate():
            input_vstreams_params = network_group.make_input_vstream_params({})
            output_vstreams_params = network_group.make_output_vstream_params({})
            
            with network_group.create_input_vstreams(input_vstreams_params) as input_vstreams, \
                 network_group.create_output_vstreams(output_vstreams_params) as output_vstreams:
                
                while True:
                    # Capture frame
                    frame_rgb = picam2.capture_array()  # 320x256 RGB
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    
                    # --- Hailo inference ---
                    t0 = time.perf_counter()
                    input_vstreams[0].send(frame_rgb)
                    hailo_output = output_vstreams[0].recv()
                    hailo_depth = hailo_output.squeeze()
                    hailo_time = time.perf_counter() - t0
                    hailo_times.append(hailo_time)
                    
                    # --- CPU inference ---
                    t0 = time.perf_counter()
                    pil_image = Image.fromarray(frame_rgb)
                    inputs = processor(images=pil_image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                    cpu_depth = outputs.predicted_depth.squeeze().cpu().numpy()
                    cpu_depth = cv2.resize(cpu_depth, (320, 256))
                    cpu_time = time.perf_counter() - t0
                    cpu_times.append(cpu_time)
                    
                    # --- Colorize both ---
                    # Hailo: lower = closer, so invert for visualization
                    hailo_color = colorize_depth(hailo_depth.max() - hailo_depth)
                    cpu_color = colorize_depth(cpu_depth)
                    
                    # --- Build comparison frame ---
                    rgb_display = frame_bgr.copy()
                    
                    # Add labels
                    cv2.putText(rgb_display, "RGB", (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    hailo_fps = 1.0 / np.mean(hailo_times[-30:]) if hailo_times else 0
                    cv2.putText(hailo_color, f"HAILO: {hailo_fps:.1f} FPS", (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(hailo_color, f"({hailo_time*1000:.1f}ms)", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    cpu_fps = 1.0 / np.mean(cpu_times[-30:]) if cpu_times else 0
                    cv2.putText(cpu_color, f"CPU: {cpu_fps:.1f} FPS", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(cpu_color, f"({cpu_time*1000:.1f}ms)", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    combined = np.hstack([rgb_display, hailo_color, cpu_color])
                    
                    cv2.imshow("Depth Comparison", combined)
                    
                    frame_count += 1
                    speedup = cpu_time / hailo_time if hailo_time > 0 else 0
                    print(f"\rFrame {frame_count} | Hailo: {hailo_time*1000:6.1f}ms | CPU: {cpu_time*1000:6.1f}ms | Speedup: {speedup:.1f}x   ", end="", flush=True)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    
    print("\n\nShutting down...")
    picam2.stop()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()