import numpy as np
import cv2

def compute_mean_color(image):
    return np.mean(image, (0, 1)).astype(float)

def compute_min_color(image):
    return np.min(image, (0, 1)).astype(float)

def compute_max_color(image):
    return np.max(image, (0, 1)).astype(float)

def main():
    # read example image from file
    image = cv2.imread("example.jpg")

    # compute mean color
    mean_color = compute_mean_color(image)
    min_color = compute_min_color(image)
    max_color = compute_max_color(image)

    # print results
    print("Mean color (BGR):", mean_color)
    print("Min color (BGR):", min_color)
    print("Max color (BGR):", max_color)

if __name__ == "__main__":
    main()
