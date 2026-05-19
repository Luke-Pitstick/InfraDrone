import numpy as np
from ultralytics.models import YOLO



class CrackDetection():
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def detect(self, image):
        return self.model.predict(image, conf=0.25, iou=0.5)

    def post_process(self, detections):
        return detections

    def run(self, image):
        return self.post_process(self.detect(image))