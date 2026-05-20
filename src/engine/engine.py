"""
This module contains the engine for the machine learning model.
"""
from .detection import DetectionEngine
from .segmentation import SegmentationEngine
from .constants import UnitTypes, DamageType, CrackSubtype, PotholeSubtype, StressRange
from .utils import display_matrix, mask_image_to_segment_arrays

class Engine:
    def __init__(self):

        self.detection_engine = DetectionEngine(model_path="models/detection.pt")
        self.segmentation_engine = SegmentationEngine(model_path="models/segmentation.pt")

    def train(self):
        pass

    def predict(self):
        pass
