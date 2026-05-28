"""
Top-level engine facade for detection and segmentation pipelines.
"""
from .detection import DetectionEngine
from .segmentation import SegmentationEngine


class Engine:
    """Coordinate detection and segmentation models for end-to-end inference."""

    def __init__(self):
        """Initialize detection and segmentation engines with default model paths."""
        self.detection_engine = DetectionEngine(model_path="models/detection.pt")
        self.segmentation_engine = SegmentationEngine(model_path="models/segmentation.pt")

    def train(self):
        """Train models (not yet implemented)."""
        pass

    def predict(self):
        """Run inference (not yet implemented)."""
        pass
