import uuid

import numpy as np
from dataclasses import dataclass
from .constants import StressRange, DamageType, CrackSubtype, PotholeSubtype, UnitTypes

class DetectionResult:
    """
    DetectionResult is a single detection for a single image.
    Images can have multiple DetectionResults.
    
    DetectionResults can have multiple SegmentationResults (pieces of damage).
    """
    
    def __init__(self, box: np.ndarray, conf: float, type: DamageType) -> None:
        # box format: x1, y1, x2, y2
        self.box: np.ndarray = box
        self.conf: float = conf
        self.type: DamageType = type
    
    def __str__(self) -> str:
        return f"DetectionResult(box: {self.box}, confidence: {self.conf}, type: {self.type})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> dict:
        return {
            "box": self.box.tolist(),
            "confidence": self.conf,
            "type": self.type
        }

class SegmentationResult:
    """
    SegmentationResult is a single mask for a single detection.
    
    Images can have multiple SegmentationResults.
    """
    def __init__(self, mask: np.ndarray, conf: float, type: DamageType) -> None:
        # Looks like 0 0.997768 0.535714 0.959821 0.535714 0.957589 0.537946...
        self.mask: np.ndarray = mask
        self.conf: float = conf
        self.type: DamageType = type
    
    def __str__(self) -> str:
        return f"SegmentationResult(mask: {self.mask}, confidence: {self.conf}, type: {self.type})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> dict:
        return {
            "mask": self.mask.tolist(),
            "confidence": self.conf,
            "type": self.type
        }
        
@dataclass
class Measurement:
    def __init__(self, value: float, unit: UnitTypes):
        self.value: float
        self.unit: UnitTypes
        
    def to_centimeters(self, value: float) -> float:
        if self.unit == UnitTypes.cm:
            return value
        elif self.unit == UnitTypes.inch:
            return value * 2.54
        else:
            raise ValueError(f"Invalid measurement unit: {self}")
        
    def to_inches(self, value: float) -> float:
        if self.unit == UnitTypes.cm:
            return value / 2.54
        elif self.unit == UnitTypes.inch:
            return value
        else:
            raise ValueError(f"Invalid measurement unit: {self}")
   
@dataclass  
class Dimensions:
    thickness: Measurement
    length: Measurement
    
    
@dataclass
class Damage:
    id: uuid.UUID
    type: DamageType
    severity: int
    confidence: float
    dimensions: Dimensions
    subtype: CrackSubtype | PotholeSubtype
    stress_range: StressRange
    num_connections: int