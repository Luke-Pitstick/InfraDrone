"""Data containers returned by the detection and segmentation engines."""

import uuid
from pathlib import Path
from typing import Literal
from datetime import datetime

import numpy as np
from dataclasses import asdict, dataclass, field
from .video import Location
from .calibration import MetricDimensions
from .motion import MotionEstimate
from .constants import StressRange, DamageType, CrackSubtype, PotholeSubtype, UnitTypes

@dataclass
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
    
    image: np.ndarray
    box: np.ndarray
    conf: float
    type: DamageType
    
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

@dataclass
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
    mask: np.ndarray
    skeleton: np.ndarray
    conf: float
    type: DamageType
    num_connections: int = 0
    endpoints: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    angle: float = 0.0
    
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
class ScalarMeasurement:
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
            "value": float(self.value),
            "unit": self.unit
        }


@dataclass
class DamageDimensions:
    """Geometric measurements for one damage instance.

    Attributes:
        thickness: Estimated average crack width or damage thickness.
        length: Estimated crack or damage length.
        area: Estimated foreground area.
    """
    thickness: ScalarMeasurement
    length: ScalarMeasurement
    area: ScalarMeasurement
    
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
    """One damage observation, not yet a defect matched across frames or surveys.

    timestamp_seconds is relative to the video's start, not wall-clock time.
    location is the phone's GPS position unless location_source is "damage".
    bounding_box is (x1, y1, x2, y2) in original-frame pixels. Dimensions use
    their declared units; area uses the square of its declared length unit.
    Unknown location, size, and severity remain None.
    """

    type: DamageType
    confidence: float
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    video_id: str | None = None
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    location: Location | None = None
    location_source: Literal["phone", "damage"] = "phone"
    dimensions: DamageDimensions | None = None
    metric_dimensions: MetricDimensions | None = None
    metric_status: str = "uncalibrated"
    triangulated_dimensions: MetricDimensions | None = None
    triangulation_status: str = "disabled"
    motion: MotionEstimate | None = None
    severity: int | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    mask_path: Path | None = None
    mask: np.ndarray | None = field(default=None, repr=False)
    skeleton: np.ndarray | None = field(default=None, repr=False)
    subtype: CrackSubtype | PotholeSubtype | None = None
    stress_range: StressRange | None = None
    num_connections: int = 0

    def to_dict(self) -> dict:
        """Return JSON-ready metadata; save mask arrays separately at mask_path."""
        return {
            "id": str(self.id),
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "type": self.type.value,
            "confidence": self.confidence,
            "location": asdict(self.location) if self.location is not None else None,
            "location_source": self.location_source if self.location is not None else None,
            "dimensions": self.dimensions.to_dict() if self.dimensions is not None else None,
            "metric_dimensions": asdict(self.metric_dimensions) if self.metric_dimensions is not None else None,
            "metric_status": self.metric_status,
            "triangulated_dimensions": asdict(self.triangulated_dimensions) if self.triangulated_dimensions is not None else None,
            "triangulation_status": self.triangulation_status,
            "motion": asdict(self.motion) if self.motion is not None else None,
            "severity": self.severity,
            "bounding_box": list(self.bounding_box) if self.bounding_box is not None else None,
            "mask_path": str(self.mask_path) if self.mask_path is not None else None,
            "subtype": self.subtype.value if self.subtype is not None else None,
            "stress_range": self.stress_range.value if self.stress_range is not None else None,
            "num_connections": self.num_connections,
        }

# Frame Models

@dataclass
class AngleMeasurement:
    """
    Angle measurement with an associated unit.
    """
    value: float
    unit: UnitTypes
    
    def to_degrees(self, value: float) -> float:
        """Convert a value from this measurement's unit to degrees.

        Args:
            value: Numeric value expressed in ``self.unit``.

        Returns:
            Converted value in degrees.
        """
        if self.unit == UnitTypes.deg:
            return value
        elif self.unit == UnitTypes.rad:
            return value * (180 / np.pi)
        else:
            raise ValueError(f"Invalid measurement unit: {self}")
    
    def to_radians(self, value: float) -> float:
        """Convert a value from this measurement's unit to radians.

        Args:
            value: Numeric value expressed in ``self.unit``.

        Returns:
            Converted value in radians.
        """
        if self.unit == UnitTypes.deg:
            return value * (np.pi / 180)
        elif self.unit == UnitTypes.rad:
            return value
        else:
            raise ValueError(f"Invalid measurement unit: {self}")
    
    def __str__(self) -> str:
        """Return a human-readable summary."""
        return f"AngleMeasurement(value: {self.value}, unit: {self.unit})"

    def __repr__(self) -> str:
        """Return the same string as :meth:`__str__`."""
        return self.__str__()

    def to_dict(self) -> dict:
        """Return a dictionary representation of the angle measurement."""
        return {
            "value": float(self.value),
            "unit": self.unit
        }

@dataclass
class Frame:
    """
    Frame object that contains the frame data and metadata.
    """
    id: str
    filepath: Path
    timestamp: datetime
    coordinates: tuple[int, int]
    processed: bool
    elevation: ScalarMeasurement
    azimuth: AngleMeasurement
    pitch: AngleMeasurement
    roll: AngleMeasurement
    yaw: AngleMeasurement
    heading: AngleMeasurement