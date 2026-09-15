"""Video decoding → segmentation → custom crack analysis → Damage observations."""

from collections.abc import Iterator
from dataclasses import dataclass, asdict
from contextlib import closing
from datetime import timedelta
import hashlib
import json
from math import isfinite
from pathlib import Path

import numpy as np

from .gps import location_at
from .models import Damage
from .motion import RoadTracker
from .segmentation import SegmentationEngine
from .video import Video, VideoFrame, Location
from .video_io import read_frames


@dataclass
class FrameResult:
    frame: VideoFrame
    location: Location | None
    damages: list[Damage]


class VideoPipeline:
    """Load one segmentation model and reuse it across frames and videos.

    PyAV accepts formats/codecs supported by its FFmpeg build, without an
    extension whitelist. Processing is incremental, one sampled frame at a time.
    Observations are not deduplicated across frames. Pixel sizes are retained;
    calibrated and optional two-view metric estimates are separate fields.
    """

    def __init__(self, model_path: Path, *, confidence_threshold: float = 0.25,
                 img_size: tuple[int, int] = (640, 640), device: str | None = None,
                 triangulation: bool = False, motion_scale_source: str = "camera_height"):
        if motion_scale_source not in ("camera_height", "gps"):
            raise ValueError("motion_scale_source must be camera_height or gps")
        self.triangulation = triangulation
        self.motion_scale_source = motion_scale_source
        self.model_path = Path(model_path)
        self.engine = SegmentationEngine(
            str(self.model_path), img_size=img_size,
            confidence_threshold=confidence_threshold, device=device,
        )

    def process_frames(self, video: Video, *, sample_fps: float | None = None,
                max_gps_gap_seconds: float = 5.0) -> Iterator[FrameResult]:
        """Yield every sampled frame with its measured observations, including empty frames.

        GPS is optional. Matching requires recorded_at and applies the video's
        gps_offset_seconds. Missing coverage stays None. Close this generator
        when stopping early to release the decoder (e.g. contextlib.closing).
        """
        if not isfinite(max_gps_gap_seconds) or max_gps_gap_seconds <= 0:
            raise ValueError("max_gps_gap_seconds must be finite and positive")
        track = sorted(video.gps, key=lambda point: point.timestamp)
        if any(a.timestamp == b.timestamp for a, b in zip(track, track[1:])):
            raise ValueError("GPS timestamps must be unique")
        if self.triangulation and video.calibration is None:
            raise ValueError("Triangulation requires video.calibration")
        tracker = RoadTracker(video.calibration, self.motion_scale_source) if self.triangulation else None
        with closing(read_frames(video, sample_fps)) as frames:
            for frame in frames:
                if video.calibration is not None:
                    video.calibration.check_shape(frame.image.shape)
                location = None
                if track and video.metadata.recorded_at is not None:
                    timestamp = video.metadata.recorded_at + timedelta(
                        seconds=frame.timestamp_seconds + video.metadata.gps_offset_seconds,
                    )
                    location = location_at(track, timestamp, max_gps_gap_seconds)
                motion = tracker.update(frame, location) if tracker is not None else None
                damages = list(self.engine.process_frame(frame.image))
                for damage in damages:
                    damage.video_id = video.id
                    damage.frame_index = frame.index
                    damage.timestamp_seconds = frame.timestamp_seconds
                    damage.location = location
                    if video.calibration is not None:
                        damage.metric_dimensions = video.calibration.measure(damage.mask)
                        damage.metric_status = "ok" if damage.metric_dimensions is not None else "outside_calibrated_region"
                    if motion is not None:
                        damage.motion = motion
                        damage.triangulation_status = motion.status
                        if motion.status == "ok":
                            damage.triangulation_status = "outside_feature_support"
                            if motion.covers(damage.bounding_box):
                                damage.triangulated_dimensions = video.calibration.measure(
                                    damage.mask, np.asarray(motion.road_normal), motion.road_distance_m,
                                )
                                damage.triangulation_status = (
                                    "ok" if damage.triangulated_dimensions is not None else "outside_calibrated_region"
                                )
                yield FrameResult(frame, location, damages)

    def process(self, video: Video, *, sample_fps: float | None = None,
                max_gps_gap_seconds: float = 5.0) -> Iterator[Damage]:
        """Flatten frame results for consumers that only need measurements."""
        with closing(self.process_frames(video, sample_fps=sample_fps,
                                        max_gps_gap_seconds=max_gps_gap_seconds)) as frames:
            for result in frames:
                yield from result.damages

    def write(self, video: Video, output_dir: Path, *, sample_fps: float | None = None,
              max_gps_gap_seconds: float = 5.0, temporal=None) -> Path:
        """Write JSONL observations and compressed masks into a new directory.

        Each mask_path is relative to output_dir and points to an NPZ containing
        mask and skeleton arrays. manifest.json records completion or failure;
        a failed run's partial outputs must not be treated as a complete survey.
        Returns the manifest path. Existing directories are never overwritten.
        """
        output_dir = Path(output_dir)
        with self.model_path.open("rb") as weights:
            model_hash = hashlib.file_digest(weights, "sha256").hexdigest()
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "masks").mkdir()
        manifest_path = output_dir / "manifest.json"
        manifest = {
            "temporal": temporal is not None,
            "triangulation": self.triangulation, "motion_scale_source": self.motion_scale_source,
            "status": "running", "video": video.to_dict(), "damage_count": 0,
            "model_sha256": model_hash, "model_path": str(self.model_path),
            "sample_fps": sample_fps, "max_gps_gap_seconds": max_gps_gap_seconds,
            "confidence_threshold": self.engine.confidence_threshold,
            "img_size": self.engine.img_size, "observations_path": "damages.jsonl",
        }
        manifest_path.write_text(json.dumps(manifest, allow_nan=False, indent=2))
        try:
            with (output_dir / "damages.jsonl").open("w") as output, closing(
                self.process_frames(video, sample_fps=sample_fps, max_gps_gap_seconds=max_gps_gap_seconds)
            ) as frames:
                for result in frames:
                    associations = ({a.observation_id: asdict(a) for a in temporal.process(video, result)}
                                    if temporal is not None else {})
                    for damage in result.damages:
                        damage.mask_path = Path("masks") / f"{damage.id}.npz"
                        np.savez_compressed(output_dir / damage.mask_path,
                                            mask=damage.mask, skeleton=damage.skeleton)
                        payload = damage.to_dict()
                        if temporal is not None:
                            payload["temporal"] = associations[str(damage.id)]
                        output.write(json.dumps(payload, allow_nan=False) + "\n")
                        manifest["damage_count"] += 1
            manifest["status"] = "complete"
        except Exception as error:
            manifest["status"] = "failed"
            manifest["error"] = str(error)
            raise
        finally:
            manifest["video"] = video.to_dict()
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, allow_nan=False, indent=2))
            temporary.replace(manifest_path)
        return manifest_path
