import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib

import cv2
import numpy as np
import threading
from picamera2 import Picamera2

Gst.init(None)

HEF_PATH    = "/usr/share/hailo-models/yolov5n_seg_h10.hef"
SO_PATH     = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolov5seg_post.so"
JSON_PATH   = "/usr/share/hailo-models/yolov5seg.json"
# Override score threshold for testing
import json, tempfile, os
with open(JSON_PATH) as f:
    cfg = json.load(f)
cfg["score_threshold"] = 0.1
_tmp_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
json.dump(cfg, _tmp_json)
_tmp_json.close()
JSON_PATH = _tmp_json.name
WIDTH, HEIGHT = 640, 640  # model expects 640x640

_lock   = threading.Lock()
_output = None  # latest rendered frame from appsink

def on_new_sample(appsink):
    global _output
    sample = appsink.emit("pull-sample")
    buf    = sample.get_buffer()
    caps   = sample.get_caps()
    w = caps.get_structure(0).get_value("width")
    h = caps.get_structure(0).get_value("height")
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if ok:
        arr = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h, w, 3).copy()
        buf.unmap(mapinfo)
        with _lock:
            _output = arr
    return Gst.FlowReturn.OK

def main():
    global _output

    # ── Camera ────────────────────────────────────────────────────────────────
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    ))
    cam.start()

    # ── GStreamer pipeline with appsrc ─────────────────────────────────────────
    pipeline_str = (
        f"appsrc name=src "
        f"  caps=video/x-raw,format=BGR,width={WIDTH},height={HEIGHT},framerate=30/1 "
        f"  stream-type=0 is-live=true block=true ! "
        f"videoconvert ! "
        f"video/x-raw,format=RGB ! "
        f"hailonet hef-path={HEF_PATH} batch-size=1 force-writable=true ! "
        f"hailofilter so-path={SO_PATH} config-path={JSON_PATH} qos=false ! "
        f"hailooverlay qos=false font-thickness=2 ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
    )

    print("Building pipeline...")
    pipeline = Gst.parse_launch(pipeline_str)

    src  = pipeline.get_by_name("src")
    sink = pipeline.get_by_name("sink")
    sink.connect("new-sample", on_new_sample)

    pipeline.set_state(Gst.State.PLAYING)
    print("Running — press 'q' to quit")

    try:
        while True:
            # Capture frame from picamera2 (RGB)
            frame = cam.capture_array()  # RGB uint8

            # Flip both axes to correct camera orientation
            frame_bgr = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), -1)
            data = frame_bgr.tobytes()
            buf   = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            src.emit("push-buffer", buf)

            # Show output if available, otherwise show raw camera feed
            with _lock:
                out = _output.copy() if _output is not None else None

            if out is not None:
                display = cv2.flip(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), -1)
            else:
                display = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), -1)

            cv2.imshow("Segmentation", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        src.emit("end-of-stream")
        pipeline.set_state(Gst.State.NULL)
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()