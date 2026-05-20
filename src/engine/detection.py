from ultralytics.models import YOLO
import numpy as np
from .models import DetectionResult, SegmentationResult
from .preprocessing import enhance_for_road_damage

class DetectionEngine():
    def __init__(self, model_path: str, img_size: tuple = (640, 640)) -> None:
        self.model = YOLO(model_path)
        self.img_size = img_size

    def detect(self, image: np.ndarray) -> DetectionResult:
        result = self.model([image])[0]

        boxes = result.boxes.numpy() if result.boxes is not None else np.array([])
        masks = result.masks.numpy() if result.masks is not None else np.array([])
        keypoints = result.keypoints.numpy() if result.keypoints is not None else np.array([])
        probs = result.probs.numpy() if result.probs is not None else np.array([])
        obb = result.obb.numpy() if result.obb is not None else np.array([])

        return DetectionResult(boxes, masks, keypoints, probs, obb)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Return a BGR uint8 image ready for YOLO inference."""
        return enhance_for_road_damage(image, size=self.img_size)

    def post_process(self, detections: DetectionResult) -> DetectionResult:
        # post-process the detections to get the final result to filter out low confidence detections
        
        
        return detections

    
    def run(self, image: np.ndarray) -> DetectionResult:
        preprocessed = self.preprocess(image)
        return self.post_process(self.detect(preprocessed))
    
    


class SegmentationEngine():
    def __init__(self, model_path: str, img_size: tuple = (640, 640)) -> None:
        self.model = YOLO(model_path)
        self.img_size = img_size

    def detect(self, image: np.ndarray) -> SegmentationResult:
        result = self.model([image])[0]

        masks = result.masks.numpy() if result.masks is not None else np.array([])
        return SegmentationResult(masks)