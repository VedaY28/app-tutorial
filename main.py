from waggle.plugin import Plugin
from waggle.data.vision import Camera
import numpy as np


def compute_mean_color(image):
    # image shape: (H, W, 3)
    return np.mean(image, axis=(0, 1)).astype(float)


def main():
    with Plugin() as plugin:
        # Try default camera (VM), fallback to SAGE camera
        try:
            camera = Camera()
        except Exception:
            camera = Camera("left")

        with camera:
            snapshot = camera.snapshot()

        mean_color = compute_mean_color(snapshot.data)

        # Waggle requires timestamps as int nanoseconds since epoch
        timestamp_ns = int(snapshot.timestamp * 1_000_000_000)

        plugin.publish("color.mean.r", float(mean_color[0]), timestamp=timestamp_ns)
        plugin.publish("color.mean.g", float(mean_color[1]), timestamp=timestamp_ns)
        plugin.publish("color.mean.b", float(mean_color[2]), timestamp=timestamp_ns)

        print("Published mean RGB:", mean_color)


if __name__ == "__main__":
    main()
