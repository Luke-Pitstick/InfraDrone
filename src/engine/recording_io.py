"""Load a recording directory into the pipeline's existing data classes."""

from contextlib import closing
import csv
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path

from .calibration import CameraCalibration
from .video import GPSPoint, Location, Video, VideoMetadata
from .video_io import read_frames


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON number: {value}")


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8-sig"), parse_constant=_reject_constant,
            parse_float=lambda value: _number(float(value), "JSON number"),
        )
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data
    except ValueError as error:
        raise ValueError(f"{path.name}: {error}") from error


def _timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value)
    if timestamp.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone")
    return timestamp.astimezone(timezone.utc)


def _number(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _load_gps(path: Path) -> list[GPSPoint]:
    if not path.exists():
        return []
    points = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"timestamp", "latitude", "longitude"}
        headers = reader.fieldnames or []
        if not required.issubset(headers) or set(headers)-required-{"accuracy_m"} or len(headers) != len(set(headers)):
            raise ValueError("gps.csv: expected timestamp,latitude,longitude[,accuracy_m] header")
        for row in reader:
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Row does not match the CSV header")
                latitude = _number(float(row["latitude"]), "latitude")
                longitude = _number(float(row["longitude"]), "longitude")
                if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                    raise ValueError("Coordinates are outside latitude/longitude bounds")
                accuracy = row.get("accuracy_m", "").strip()
                accuracy = _number(float(accuracy), "accuracy_m", positive=True) if accuracy else None
                points.append(GPSPoint(_timestamp(row["timestamp"]), Location(latitude, longitude, accuracy)))
            except (TypeError, ValueError) as error:
                raise ValueError(f"gps.csv row {reader.line_num}: {error}") from error
    points.sort(key=lambda point: point.timestamp)
    if any(a.timestamp == b.timestamp for a, b in zip(points, points[1:])):
        raise ValueError("gps.csv: duplicate GPS timestamps")
    return points


def load_video(recording_directory: Path) -> Video:
    """Load one video.* file plus optional metadata.json, gps.csv, calibration.json.

    Decode only the first frame to validate the video and calibration dimensions;
    later corruption is reported during processing. Missing sidecars are allowed,
    but malformed sidecars are errors. GPS requires recorded_at for clock matching.
    Metadata is flat: VideoMetadata fields plus optional id and route_id.
    """
    directory = Path(recording_directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    footage = sorted(path for path in directory.glob("video.*") if path.is_file())
    if len(footage) != 1:
        raise ValueError("Recording directory must contain exactly one video.* file")
    data = _read_json(directory / "metadata.json") if (directory / "metadata.json").exists() else {}
    try:
        identifiers = {key: data.pop(key) for key in ("id", "route_id") if key in data}
        for key, value in identifiers.items():
            if key == "route_id" and value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a nonempty string")
        if data.get("recorded_at") is not None:
            data["recorded_at"] = _timestamp(data["recorded_at"])
        for key in ("width", "height"):
            if data.get(key) is not None and (type(data[key]) is not int or data[key] <= 0):
                raise ValueError(f"{key} must be a positive integer")
        for key in ("fps", "duration_seconds", "gps_offset_seconds"):
            if key in data and (data[key] is not None or key == "gps_offset_seconds"):
                _number(data[key], key, positive=key != "gps_offset_seconds")
        for key in ("camera_id", "calibration_reason"):
            if data.get(key) is not None and not isinstance(data[key], str):
                raise ValueError(f"{key} must be a string")
        if data.get("clock_mapping") is not None and not isinstance(data["clock_mapping"], dict):
            raise ValueError("clock_mapping must be an object")
        if data.get("calibration_status") not in (None, "available", "unavailable"):
            raise ValueError("calibration_status must be available or unavailable")
        metadata = VideoMetadata(**data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata.json: {error}") from error

    calibration = None
    calibration_path = directory / "calibration.json"
    if calibration_path.exists():
        try:
            values = _read_json(calibration_path)
            for key, value in values.items():
                if key in ("image_width", "image_height"):
                    if type(value) is not int or value <= 0:
                        raise ValueError(f"{key} must be a positive integer")
                elif key in ("road_roi", "distortion"):
                    if key == "road_roi" and value is None:
                        continue
                    if not isinstance(value, list):
                        raise ValueError(f"{key} must be an array")
                    for item in value:
                        _number(item, key)
                        if key == "road_roi" and type(item) is not int:
                            raise ValueError("road_roi must contain integers")
                else:
                    _number(value, key)
            calibration = CameraCalibration(**values)
        except (TypeError, ValueError) as error:
            raise ValueError(f"calibration.json: {error}") from error
    status = "available" if calibration is not None else "unavailable"
    if metadata.calibration_status is not None and metadata.calibration_status != status:
        raise ValueError("metadata.json: calibration_status conflicts with calibration.json presence")
    metadata.calibration_status = status
    if calibration is None and metadata.calibration_reason is None:
        metadata.calibration_reason = "No calibration.json supplied"
    gps = _load_gps(directory / "gps.csv")
    if gps and metadata.recorded_at is None:
        raise ValueError("metadata.json: recorded_at is required to synchronize supplied GPS samples")
    video = Video(footage[0], metadata=metadata, gps=gps, calibration=calibration, **identifiers)
    declared_width, declared_height = metadata.width, metadata.height
    with closing(read_frames(video)) as frames:
        frame = next(frames)
        height, width = frame.image.shape[:2]
        if ((declared_width is not None and declared_width != width)
                or (declared_height is not None and declared_height != height)):
            raise ValueError("metadata.json: dimensions do not match the encoded video")
        metadata.width, metadata.height = width, height
        if calibration is not None:
            calibration.check_shape(frame.image.shape)
    return video
