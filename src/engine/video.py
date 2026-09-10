"""Small data containers for a video and its timestamped GPS track."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from .calibration import CameraCalibration


@dataclass
class VideoFrame:
    """Decoded BGR pixels and their position in the source video.

    index counts every decoded frame, including frames skipped by sampling.
    timestamp_seconds uses presentation time relative to the first frame.
    Image height and width are available through image.shape[:2].
    """

    image: np.ndarray = field(repr=False)
    index: int
    timestamp_seconds: float


@dataclass
class Location:
    """WGS84 coordinates; accuracy is the reported horizontal accuracy in meters."""

    latitude: float
    longitude: float
    accuracy_m: float | None = None


@dataclass
class GPSPoint:
    """A phone location at an absolute, timezone-aware GPS timestamp."""

    timestamp: datetime
    location: Location

    def __post_init__(self) -> None:
        if self.timestamp.utcoffset() is None:
            raise ValueError("GPS timestamps must include a timezone")


@dataclass
class VideoMetadata:
    """Recording properties, populated when the footage is inspected.

    Frame times are seconds from the start of the video. Add them and
    gps_offset_seconds to recorded_at to obtain the corresponding GPS time.
    fps is descriptive only; use frame presentation timestamps for sampling.
    """

    recorded_at: datetime | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    gps_offset_seconds: float = 0.0
    camera_id: str | None = None
    calibration_status: str | None = None
    calibration_reason: str | None = None
    clock_mapping: dict | None = None

    def __post_init__(self) -> None:
        if self.recorded_at is not None and self.recorded_at.utcoffset() is None:
            raise ValueError("Recording start time must include a timezone")


@dataclass
class Video:
    """One survey recording. Footage stays on disk; GPS samples stay in memory.

    route_id groups recordings of the same route across separate surveys.
    GPS loading and matching are separate operations, not constructor side effects.
    """

    footage_path: Path
    id: str = field(default_factory=lambda: str(uuid4()))
    gps: list[GPSPoint] = field(default_factory=list)
    metadata: VideoMetadata = field(default_factory=VideoMetadata)
    route_id: str | None = None
    calibration: CameraCalibration | None = None

    def to_dict(self) -> dict:
        """Return a JSON-ready manifest referencing, rather than embedding, footage."""
        metadata = asdict(self.metadata)
        metadata["recorded_at"] = (
            self.metadata.recorded_at.isoformat()
            if self.metadata.recorded_at is not None else None
        )
        return {
            "id": self.id,
            "footage_path": str(self.footage_path),
            "route_id": self.route_id,
            "calibration": asdict(self.calibration) if self.calibration is not None else None,
            "metadata": metadata,
            "gps": [
                {"timestamp": point.timestamp.isoformat(), "location": asdict(point.location)}
                for point in self.gps
            ],
        }
