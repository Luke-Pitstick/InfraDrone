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
    def __init__(self, mask: np.ndarray, skeleton: np.ndarray, conf: float, type: DamageType, endpoints: np.ndarray = np.array([]), angle: float = 0.0) -> None:
        self.mask: np.ndarray = mask
        self.skeleton: np.ndarray = skeleton
        self.conf: float = conf
        self.type: DamageType = type
        self.endpoints: np.ndarray = endpoints
        self.angle: float = angle
        
    
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
    value: float
    unit: UnitTypes
        
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
        
    def __str__(self) -> str:
        return f"Measurement(value: {self.value}, unit: {self.unit})"
    
    def __repr__(self) -> str:
        return self.__str__()
        
    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit
        }
    
   
@dataclass  
class Dimensions:
    thickness: Measurement
    length: Measurement
    
    def __str__(self) -> str:
        return f"Dimensions(thickness: {self.thickness}, length: {self.length})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> dict:
        return {
            "thickness": self.thickness.to_dict(),
            "length": self.length.to_dict()
        }
    
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
    
    def __str__(self) -> str:
        return f"Damage(id: {self.id}, type: {self.type}, severity: {self.severity}, confidence: {self.confidence}, dimensions: {self.dimensions}, subtype: {self.subtype}, stress_range: {self.stress_range}, num_connections: {self.num_connections})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "severity": self.severity,
            "confidence": self.confidence,
            "dimensions": self.dimensions.to_dict(),
            "subtype": self.subtype,
            "stress_range": self.stress_range,
            "num_connections": self.num_connections
        }