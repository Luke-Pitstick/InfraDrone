"""Data containers returned by the detection and segmentation engines."""

import uuid

import numpy as np
from dataclasses import dataclass
from .constants import StressRange, DamageType, CrackSubtype, PotholeSubtype, UnitTypes

class DetectionResult:
    """Bounding-box detection produced for one image.

    A source image can produce multiple detections, and each detection can later
    be refined into one or more segmented damage regions.

    Attributes:
        image: Source image associated with the detection.
        box: Bounding box as ``[x1, y1, x2, y2]`` in pixel coordinates.
        conf: Detection confidence score.
        type: Detected road-damage type.
    """
    
    def __init__(self, image: np.ndarray, box: np.ndarray, conf: float, type: DamageType) -> None:
        """Store one bounding-box detection.

        Args:
            image: Source image (full frame or crop).
            box: ``[x1, y1, x2, y2]`` in pixel coordinates.
            conf: Detection confidence in ``[0, 1]``.
            type: Damage class for this box.
        """
        # box format: x1, y1, x2, y2
        self.image: np.ndarray = image
        self.box: np.ndarray = box
        self.conf: float = conf
        self.type: DamageType = type
    
    def __str__(self) -> str:
        """Return a human-readable summary."""
        return f"DetectionResult(box: {self.box}, confidence: {self.conf}, type: {self.type})"

    def __repr__(self) -> str:
        """Return the same string as :meth:`__str__`."""
        return self.__str__()

    def to_dict(self) -> dict:
        """Return a dictionary representation of the detection."""
        return {
            "box": self.box.tolist(),
            "confidence": self.conf,
            "type": self.type
        }

class SegmentationResult:
    """Segmentation mask and skeleton for one damage instance.

    Attributes:
        mask: Binary foreground mask for the detected damage.
        skeleton: Skeletonized version of ``mask``.
        conf: Segmentation confidence score.
        type: Detected road-damage type.
        num_connections: Count of detected skeleton junctions.
        endpoints: Skeleton endpoint coordinates as ``[row, col]`` pairs.
        angle: Estimated branch orientation in degrees.
    """
    def __init__(self, mask: np.ndarray, skeleton: np.ndarray, conf: float, type: DamageType, num_connections: int = 0, endpoints: np.ndarray = np.array([]), angle: float = 0.0) -> None:
        """Store one segmented damage instance.

        Args:
            mask: Binary segmentation mask.
            skeleton: 1-pixel-wide skeleton of the mask.
            conf: Model confidence for this instance.
            type: Damage class (crack or pothole).
            num_connections: Junction count along the skeleton.
            endpoints: ``(N, 2)`` array of ``[row, col]`` skeleton endpoints.
            angle: Branch axis angle in degrees.
        """
        self.mask: np.ndarray = mask
        self.skeleton: np.ndarray = skeleton
        self.conf: float = conf
        self.type: DamageType = type
        self.num_connections: int = num_connections
        self.endpoints: np.ndarray = endpoints
        self.angle: float = angle
        
        
    
    def __str__(self) -> str:
        """Return a human-readable summary."""
        return f"SegmentationResult(mask: {self.mask}, confidence: {self.conf}, type: {self.type})"

    def __repr__(self) -> str:
        """Return the same string as :meth:`__str__`."""
        return self.__str__()

    def to_dict(self) -> dict:
        """Return a dictionary representation of the segmentation result."""
        return {
            "mask": self.mask.tolist(),
            "confidence": self.conf,
            "type": self.type
        }

@dataclass
class Measurement:
    """Scalar measurement with an associated unit.

    Attributes:
        value: Numeric measurement value.
        unit: Unit used for ``value``.
    """
    value: float
    unit: UnitTypes
        
    def to_centimeters(self, value: float) -> float:
        """Convert a value from this measurement's unit to centimeters.

        Args:
            value: Numeric value expressed in ``self.unit``.

        Returns:
            Converted value in centimeters.

        Raises:
            ValueError: If ``self.unit`` cannot be converted to centimeters.
        """
        if self.unit == UnitTypes.cm:
            return value
        elif self.unit == UnitTypes.inch:
            return value * 2.54
        else:
            raise ValueError(f"Invalid measurement unit: {self}")
        
    def to_inches(self, value: float) -> float:
        """Convert a value from this measurement's unit to inches.

        Args:
            value: Numeric value expressed in ``self.unit``.

        Returns:
            Converted value in inches.

        Raises:
            ValueError: If ``self.unit`` cannot be converted to inches.
        """
        if self.unit == UnitTypes.cm:
            return value / 2.54
        elif self.unit == UnitTypes.inch:
            return value
        else:
            raise ValueError(f"Invalid measurement unit: {self}")
        
    def __str__(self) -> str:
        """Return a human-readable summary."""
        return f"Measurement(value: {self.value}, unit: {self.unit})"

    def __repr__(self) -> str:
        """Return the same string as :meth:`__str__`."""
        return self.__str__()

    def to_dict(self) -> dict:
        """Return a dictionary representation of the measurement."""
        return {
            "value": self.value,
            "unit": self.unit
        }


@dataclass
class Dimensions:
    """Geometric measurements for one damage instance.

    Attributes:
        thickness: Estimated average crack width or damage thickness.
        length: Estimated crack or damage length.
        area: Estimated foreground area.
    """
    thickness: Measurement
    length: Measurement
    area: Measurement
    
    def __str__(self) -> str:
        """Return a human-readable summary."""
        return f"Dimensions(thickness: {self.thickness}, length: {self.length}, area: {self.area})"

    def __repr__(self) -> str:
        """Return the same string as :meth:`__str__`."""
        return self.__str__()

    def to_dict(self) -> dict:
        """Return a dictionary representation of all dimensions."""
        return {
            "thickness": self.thickness.to_dict(),
            "length": self.length.to_dict(),
            "area": self.area.to_dict()
        }

@dataclass
class Damage:
    """Fully analyzed road-damage instance.

    Attributes:
        id: Unique identifier for the damage record.
        mask: Binary mask for the damage region.
        skeleton: Skeletonized mask used for branch measurements.
        type: High-level damage type.
        severity: Computed or assigned severity score.
        confidence: Detection or segmentation confidence score.
        dimensions: Thickness, length, and area measurements.
        subtype: Crack or pothole subtype classification.
        stress_range: Road-class stress factor used in severity analysis.
        num_connections: Number of detected skeleton branch connections.
    """
    id: uuid.UUID
    mask: np.ndarray
    skeleton: np.ndarray
    type: DamageType
    severity: int
    confidence: float
    dimensions: Dimensions
    subtype: CrackSubtype | PotholeSubtype
    stress_range: StressRange
    num_connections: int
    
    def __str__(self) -> str:
        """Return a human-readable summary."""
        return f"Damage(id: {self.id}, type: {self.type}, severity: {self.severity}, confidence: {self.confidence}, dimensions: {self.dimensions}, subtype: {self.subtype}, stress_range: {self.stress_range}, num_connections: {self.num_connections})"

    def __repr__(self) -> str:
        """Return the same string as :meth:`__str__`."""
        return self.__str__()

    def to_dict(self) -> dict:
        """Return a dictionary representation of the damage record."""
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
