from pathlib import Path
import argparse

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


# Dataset yaml lives in python/datasets/yolo_pics/ (path inside yaml is relative to that directory).
_PYTHON_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_YAML = _PYTHON_ROOT / "datasets" / "yolo_pics" / "converted.yaml"



def select_image():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file")
    args = parser.parse_args()
    return args.image_path

class DamageDetectionModel:
    def __init__(self):
        self.model = YOLO("../models/trained_models/best.pt")

    def detect(self, image_path):
        results = self.model(image_path)
        return results
    
    

if __name__ == "__main__":
    model = DamageDetectionModel()
    image_path = select_image()
    results = model.detect(image_path)
    annotated_img = results[0].plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    plt.imshow(annotated_img_rgb)
    plt.axis("off")
    plt.show()