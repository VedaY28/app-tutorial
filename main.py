import argparse
import os
from ultralytics import YOLO
import time
from waggle.plugin import Plugin

def run_inference(file_path: str, batch_size: int, device: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "firedetect-11s.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = YOLO(model_path)
    results = model.predict(source=file_path, batch=batch_size, device=device, stream=True, imgsz=640, conf=0.25, verbose=False)
    
    with Plugin() as plugin:
        for i, result in enumerate(results):
            ts = int(time.time() * 1_000_000_000)
            detections = len(result.boxes)
            inference_time = result.speed.get("inference", 0)

            plugin.publish("result.frame", i, timestamp=ts)
            plugin.publish("result.detections", detections, timestamp=ts)
            plugin.publish("inference.time.ms", inference_time, timestamp=ts)

            if detections > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])

                    plugin.publish("object.label", label, timestamp=ts)
                    plugin.publish("object.confidence", conf, timestamp=ts)
    print("Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 Inference Script")
    parser.add_argument("--file", type=str, required=True, help="Path to image or video")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images per batch")
    parser.add_argument("--device", type=str, default="0", help="Device to run on: '0', '1' or 'cpu'")
    
    args = parser.parse_args()
    run_inference(args.file, args.batch_size, args.device)
