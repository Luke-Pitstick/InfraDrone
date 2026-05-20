from ultralytics.models import YOLO
import numpy as np

from .preprocessing import enhance_for_road_damage
from .models import DetectionResult
from .base import BaseEngine
from .constants import UnitTypes, DamageType


CLASS_MAP = {
    0: DamageType.CRACK,
    1: DamageType.POTHOLE
}

class DetectionEngine(BaseEngine):
    def __init__(self, model_path: str, img_size: tuple = (640, 640)) -> None:
        super().__init__(unit_type=UnitTypes.cm, width_ratio=1, height_ratio=1)
        self.model = YOLO(model_path)
        self.img_size = img_size

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        result = self.model([image], imgsz=self.img_size)[0].cpu().numpy()

        detection_results = []

        if result.boxes is None:
            return detection_results

        for i, box in enumerate(result.boxes.data):
            # box format: x1, y1, x2, y2, confidence, class
            xyxy = box[:4]
            conf = float(box[4])
            cls = int(box[5])
            class_type = CLASS_MAP[cls]


            detection_results.append(
                DetectionResult(
                    xyxy,
                    conf,
                    class_type,
                )
            )

        return detection_results

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Return a BGR uint8 image ready for YOLO inference."""
        return enhance_for_road_damage(image, size=self.img_size)

    def post_process(self, detections: list[DetectionResult]) -> list[DetectionResult]:
        # post-process the detections to get the final result to filter out low confidence detections
        
        
        return detections
    
    # Returns an np.ndarray of each detection cropped from the original image
    def crop_images(self, image: np.ndarray, detections: list[DetectionResult]) -> list[np.ndarray]:
        return [image[detection.box[1]:detection.box[3], detection.box[0]:detection.box[2]] for detection in detections]

    
    def run(self, image: np.ndarray) -> list[DetectionResult]:
        preprocessed = self.preprocess(image)
        return self.post_process(self.detect(preprocessed))
    
    
