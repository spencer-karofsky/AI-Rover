#!/usr/bin/env python3
"""
Side-by-side comparison: Depth-Anything-V2 (CPU) vs SCDepthV3 (Hailo)
"""
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from picamera2 import Picamera2
from libcamera import Transform
from hailo_platform import HEF, VDevice, HailoSchedulingAlgorithm
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Shared state
latest_frame = None
frame_lock = threading.Lock()
running = True


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


def start_web_server():
    """Simple MJPEG stream server"""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(b'''<html><body style="margin:0;background:#000;">
                    <img src="/stream" style="width:100%;max-width:1200px;">
                    </body></html>''')
            elif self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                while running:
                    with frame_lock:
                        if latest_frame is not None:
                            _, jpg = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                            self.wfile.write(jpg.tobytes())
                            self.wfile.write(b'\r\n')
                    time.sleep(0.03)
        def log_message(self, *args): pass
    
    server = HTTPServer(('0.0.0.0', 8080), Handler)
    server.timeout = 1
    while running:
        server.handle_request()


def main():
    global latest_frame, running
    
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
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.group_id = "SHARED"
    vdevice = VDevice(params)
    infer_model = vdevice.create_infer_model(hef_path)
    infer_model.set_batch_size(1)
    configured_model = infer_model.configure()
    hailo_output_shape = infer_model.output().shape
    print(f"    Hailo input: {infer_model.input().shape}, output: {hailo_output_shape}")
    
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
    
    # --- Start web server ---
    print("[4/4] Starting web server...")
    threading.Thread(target=start_web_server, daemon=True).start()
    
    print("=" * 50)
    print("Stream: http://<your-pi-ip>:8080")
    print("Press Ctrl+C to quit")
    print("=" * 50)
    
    # Timing trackers
    hailo_times = []
    cpu_times = []
    frame_count = 0
    
    try:
        while running:
            # Capture frame
            frame_rgb = picam2.capture_array()  # 320x256 RGB
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # --- Hailo inference ---
            t0 = time.perf_counter()
            output_buffer = np.empty(hailo_output_shape, dtype=np.uint16)
            bindings = configured_model.create_bindings(
                output_buffers={infer_model.output().name: output_buffer}
            )
            bindings.input().set_buffer(frame_rgb)
            configured_model.run([bindings], timeout=1000)
            hailo_depth = output_buffer.squeeze()
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
            # RGB | Hailo Depth | CPU Depth
            rgb_small = cv2.resize(frame_bgr, (320, 256))
            
            # Add labels
            cv2.putText(rgb_small, "RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
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
            
            combined = np.hstack([rgb_small, hailo_color, cpu_color])
            
            # Update shared frame for web server
            with frame_lock:
                latest_frame = combined.copy()
            
            frame_count += 1
            speedup = cpu_time / hailo_time if hailo_time > 0 else 0
            print(f"\rFrame {frame_count} | Hailo: {hailo_time*1000:6.1f}ms | CPU: {cpu_time*1000:6.1f}ms | Speedup: {speedup:.1f}x   ", end="", flush=True)
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        running = False
        picam2.stop()
        print("Done.")


if __name__ == '__main__':
    main()