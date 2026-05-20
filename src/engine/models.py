import numpy as np


class DetectionResult():
    def __init__(self, boxes: np.ndarray, masks: np.ndarray, keypoints: np.ndarray, probs: np.ndarray, obb: np.ndarray) -> None:
        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints
        self.probs = probs
        self.obb = obb
    
    def __str__(self) -> str:
        return f"DetectionResult(boxes: {self.boxes}, masks: {self.masks}, keypoints: {self.keypoints}, probs: {self.probs}, obb: {self.obb})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> dict:
        return {
            "boxes": self.boxes.tolist(),
            "masks": self.masks.tolist(),
            "keypoints": self.keypoints.tolist(),
            "probs": self.probs.tolist(),
            "obb": self.obb.tolist()
        }


class SegmentationResult():
    def __init__(self, masks: np.ndarray) -> None:
        self.masks = masks
        
    
    def __str__(self) -> str:
        return f"SegmentationResult(masks: {self.masks})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> dict:
        return {
            "masks": self.masks.tolist()
        }