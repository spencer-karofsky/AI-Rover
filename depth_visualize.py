#!/usr/bin/env python3
"""
Side-by-side comparison: Depth-Anything-V2 (CPU) vs SCDepthV3 (Hailo)
"""
import numpy as np
import cv2
import time
import os
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from picamera2 import Picamera2
from hailo_platform import VDevice, HailoSchedulingAlgorithm

os.environ['DISPLAY'] = ':0'

def colorize_depth_hailo(depth):
    """Colorize Hailo depth (lower = closer)"""
    d = depth.squeeze().astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    d = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)

def colorize_depth_cpu(depth):
    """Colorize CPU depth (higher = closer typically)"""
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    d = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)

def main():
    print("=" * 50)
    print("DEPTH COMPARISON: CPU vs HAILO")
    print("=" * 50)
    
    # --- Camera setup ---
    print("[1/3] Starting camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 256), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    
    # --- Hailo setup ---
    print("[2/3] Loading Hailo model (SCDepthV3)...")
    hef_path = '/home/spencer/ai_rover/scdepthv3.hef'
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.group_id = "SHARED"
    vdevice = VDevice(params)
    infer_model = vdevice.create_infer_model(hef_path)
    infer_model.set_batch_size(1)
    print(f"    Input: {infer_model.input().shape}, Output: {infer_model.output().shape}")
    
    # --- CPU model setup ---
    print("[3/3] Loading CPU model (Depth-Anything-V2)...")
    processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf",
        use_fast=True
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    model.eval()
    
    print("=" * 50)
    print("Press 'q' to quit")
    print("=" * 50)
    
    hailo_times = []
    cpu_times = []
    frame_count = 0
    
    with infer_model.configure() as configured_model:
        while True:
            # Capture frame
            frame = cv2.flip(picam2.capture_array(), 0)
            frame_rgb = frame
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # --- Hailo inference ---
            t0 = time.perf_counter()
            output_buffer = np.empty(infer_model.output().shape, dtype=np.uint16)
            bindings = configured_model.create_bindings(
                output_buffers={infer_model.output().name: output_buffer}
            )
            bindings.input().set_buffer(frame_rgb)
            configured_model.run([bindings], timeout=1000)
            hailo_depth = output_buffer
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
            
            # --- Colorize ---
            hailo_color = colorize_depth_hailo(hailo_depth)
            cpu_color = colorize_depth_cpu(cpu_depth)
            
            # --- Add labels ---
            hailo_fps = 1.0 / np.mean(hailo_times[-30:]) if hailo_times else 0
            cpu_fps = 1.0 / np.mean(cpu_times[-30:]) if cpu_times else 0
            
            cv2.putText(frame_bgr, "RGB", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(hailo_color, f"HAILO {hailo_fps:.1f}fps", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(hailo_color, f"{hailo_time*1000:.1f}ms", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(cpu_color, f"CPU {cpu_fps:.1f}fps", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(cpu_color, f"{cpu_time*1000:.1f}ms", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # --- Combine and show ---
            combined = np.hstack([frame_bgr, hailo_color, cpu_color])
            
            frame_count += 1
            speedup = cpu_time / hailo_time if hailo_time > 0 else 0
            print(f"\rFrame {frame_count} | Hailo: {hailo_time*1000:5.1f}ms | CPU: {cpu_time*1000:6.1f}ms | {speedup:.0f}x faster   ", end="", flush=True)
            
            cv2.imshow("Depth Comparison", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    print("\n\nShutting down...")
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()