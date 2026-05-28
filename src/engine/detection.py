from ultralytics.models import YOLO
import numpy as np
import cv2

from .preprocessing import enhance_for_road_damage
from .models import DetectionResult
from .base import BaseEngine
from .constants import UnitTypes, DamageType


CLASS_MAP = {
    0: DamageType.CRACK,
    1: DamageType.POTHOLE
}


class DetectionEngine(BaseEngine):
    """Run YOLO object detection for road damage (cracks and potholes)."""

    def __init__(self, model_path: str, img_size: tuple = (640, 640)) -> None:
        """Load a YOLO detection model and configure output units.

        Args:
            model_path: Path to the YOLO weights file.
            img_size: Inference size passed to YOLO as ``imgsz``.
        """
        super().__init__(unit_type=UnitTypes.cm, width_ratio=1, height_ratio=1)
        self.model = YOLO(model_path)
        self.img_size = img_size

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """Run YOLO on a single image and return bounding-box detections.

        Args:
            image: BGR image array.

        Returns:
            One ``DetectionResult`` per box, each holding the full image and xyxy box.
        """
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
                    image,
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
        """Filter detections by confidence threshold.

        Args:
            detections: Raw model outputs.

        Returns:
            Detections with ``conf > 0.5``.
        """
        detections = [detection for detection in detections if detection.conf > 0.5]
        return detections

    def crop_images(self, detections: list[DetectionResult]) -> list[DetectionResult]:
        """Crop each detection's image to its bounding box in place.

        Args:
            detections: Detections with ``image`` and ``box`` (x1, y1, x2, y2).

        Returns:
            The same list with ``detection.image`` replaced by the crop.
        """
        for detection in detections:
            detection.image = detection.image[
                detection.box[1] : detection.box[3],
                detection.box[0] : detection.box[2],
                :,
            ]
        return detections

    def process_frame(self, image: np.ndarray) -> list[DetectionResult]:
        """Preprocess, detect, crop, and filter detections for one image.

        Args:
            image: Raw BGR input image.

        Returns:
            Final detection results ready for downstream use.
        """
        preprocessed = self.preprocess(image)
        detections = self.detect(preprocessed)
        detections = self.post_process(detections)
        detections = self.crop_images(detections)

        return detections

    def _process_frame_test(self, image: np.ndarray) -> list[DetectionResult]:
        """Detect and filter without cropping (for quick tests).

        Args:
            image: Raw BGR input image.

        Returns:
            Filtered detections on the full preprocessed image.
        """
        preprocessed = self.preprocess(image)
        return self.post_process(self.detect(preprocessed))


if __name__ == "__main__":
    MODEL_PATH = "/Users/lukepitstick/Projects/Data-Science/InfraDrone/src/ml/models/weights/yolo26s-seg.pt"
    IMAGE_PATH = "/Users/lukepitstick/Projects/Data-Science/InfraDrone/datasets/detection/RD2022/test/images/China_Drone_000401.jpg"

    engine = DetectionEngine(MODEL_PATH)
    image = cv2.imread(IMAGE_PATH)
    detections = engine._process_frame_test(image)
    print(detections)
