import argparse
import os
import time
from ultralytics import YOLO
from waggle.plugin import Plugin

def run_yolo(file_path: str, batch_size: int, device: str):
    """
    Run YOLO inference on an image or video file and publish results to Sage.
    Publishes frame number, fire count, smoke count, and inference time per frame.
    """

    # Load model from same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "firedetect-11s.pt")
    model = YOLO(model_path)

    # Stream predictions
    results = model.predict(file_path, batch=batch_size, stream=True, imgsz=640, device=device, verbose=False)

    with Plugin() as plugin:
        frame_number = 0
        total_fires = 0
        total_smoke = 0

        for i, result in enumerate(results):
            # Only inference time
            inference_time = result.speed.get("inference", 0)

            # Count objects by class (class 0=fire, class 1=smoke)
            if hasattr(result, "boxes") and len(result.boxes) > 0:
                cls_ids = result.boxes.cls
                fire_count = int((cls_ids == 0).sum())
                smoke_count = int((cls_ids == 1).sum())
            else:
                fire_count = 0
                smoke_count = 0

            ts = int(time.time() * 1_000_000_000)

            # Publish per-frame data
            plugin.publish("frame.number", i, timestamp=ts)
            plugin.publish("fire.count", fire_count, timestamp=ts)
            plugin.publish("smoke.count", smoke_count, timestamp=ts)
            plugin.publish("inference.time.ns", inference_time, timestamp=ts)

            # Update totals
            total_fires += fire_count
            total_smoke += smoke_count
            frame_number += 1

        ts_total = int(time.time() * 1_000_000_000)

        # Publish totals
        plugin.publish("fire.total", total_fires, timestamp=ts_total)
        plugin.publish("smoke.total", total_smoke, timestamp=ts_total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to image or video file for YOLO inference")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for YOLO inference")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index (0,1,...) or 'cpu'")
    args = parser.parse_args()

    run_yolo(args.file, batch_size=args.batch_size, device=args.device)
