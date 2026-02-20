import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import VDevice, HailoSchedulingAlgorithm

HEF_PATH = "/usr/share/hailo-models/yolov8m_h10.hef"
RESOLUTION = (640, 480)
CONF_THRESH = 0.5

COCO_LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

def main():
    # ── Camera ────────────────────────────────────────────────────────────────
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": RESOLUTION, "format": "RGB888"}
    ))
    cam.start()

    # ── Hailo ─────────────────────────────────────────────────────────────────
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    vdevice = VDevice(params)

    infer_model = vdevice.create_infer_model(HEF_PATH)
    infer_model.set_batch_size(1)
    configured = infer_model.configure()

    input_h = infer_model.input().shape[0]
    input_w = infer_model.input().shape[1]
    print(f"Model input: {input_h}x{input_w}")
    print("Running — press 'q' to quit")

    while True:
        # Picamera2 with RGB888 gives RGB directly — no conversion needed
        frame = cam.capture_array()  # RGB uint8
        orig_h, orig_w = frame.shape[:2]

        # Convert BGR→RGB for model (YOLOv8 trained on RGB)
        input_rgb = cv2.cvtColor(cv2.resize(frame, (input_w, input_h)), cv2.COLOR_BGR2RGB)
        input_data = input_rgb.astype(np.uint8)

        # Infer
        output_buffers = {
            o.name: np.empty(infer_model.output(o.name).shape, dtype=np.float32)
            for o in infer_model.outputs
        }
        bindings = configured.create_bindings(output_buffers=output_buffers)
        bindings.input().set_buffer(input_data)
        configured.run([bindings], timeout=1000)

        # Frame is already BGR from picamera2, just flip both axes
        display = cv2.flip(frame, -1)

        # Parse Hailo NMS output and debug first firing detection
        _debug_printed = getattr(main, '_debug_printed', False)
        for name, buf in output_buffers.items():
            flat = buf.flatten()
            n = len(flat) // 6
            detections = flat[:n * 6].reshape(n, 6)

            for det in detections:
                cls_id, score, y_min, x_min, y_max, x_max = det
                if score < CONF_THRESH:
                    continue

                if not _debug_printed:
                    print(f"Raw detection: {det}")
                    main._debug_printed = True

                x1 = int(x_min * orig_w)
                y1 = int(y_min * orig_h)
                x2 = int(x_max * orig_w)
                y2 = int(y_max * orig_h)

                # Mirror x coords to match flipped display
                x1_m = orig_w - x2
                x2_m = orig_w - x1
                # Mirror y coords too (flipped on both axes)
                y1_m = orig_h - y2
                y2_m = orig_h - y1

                cls_id = int(cls_id)
                label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else str(cls_id)
                cv2.rectangle(display, (x1_m, y1_m), (x2_m, y2_m), (0, 255, 0), 2)
                cv2.putText(display, f"{label} {score:.2f}",
                            (x1_m, max(y1_m - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()