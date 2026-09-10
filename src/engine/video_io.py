"""Incremental video decoding with presentation-time sampling."""

from collections.abc import Iterator
from fractions import Fraction
from math import isfinite

import av

from .video import Video, VideoFrame


def read_frames(video: Video, sample_fps: float | None = None) -> Iterator[VideoFrame]:
    """Yield BGR frames and populate available video stream metadata.

    With sample_fps, yield the first frame at or after each sampling boundary;
    gaps never duplicate frames. None yields every frame. Timestamps are
    relative to the first decoded frame, which recorded_at should describe.
    Missing or decreasing presentation timestamps raise ValueError rather than
    guessing from FPS. Pixels retain the encoded orientation (no rotation).

    The file closes on exhaustion or generator.close(). When stopping early,
    use contextlib.closing(read_frames(...)) to close it deterministically.
    """
    if sample_fps is not None and (not isfinite(sample_fps) or sample_fps <= 0):
        raise ValueError("sample_fps must be a finite positive number")
    interval = Fraction(1, 1) / Fraction(str(sample_fps)) if sample_fps is not None else None
    next_sample = Fraction(0)
    origin = None
    previous = None

    with av.open(str(video.footage_path)) as container:
        if not container.streams.video:
            raise ValueError("File contains no video stream")
        stream = container.streams.video[0]
        video.metadata.width = stream.codec_context.width
        video.metadata.height = stream.codec_context.height
        video.metadata.fps = float(stream.average_rate) if stream.average_rate else None
        video.metadata.duration_seconds = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None else None
        )

        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Frame {index} has no presentation timestamp")
            presentation_time = frame.pts * frame.time_base
            if previous is not None and presentation_time < previous:
                raise ValueError(f"Frame {index} has a decreasing presentation timestamp")
            previous = presentation_time
            if origin is None:
                origin = presentation_time
            timestamp = presentation_time - origin
            if interval is not None:
                if timestamp < next_sample:
                    continue
                next_sample = (timestamp // interval + 1) * interval
            yield VideoFrame(
                image=frame.to_ndarray(format="bgr24"),
                index=index,
                timestamp_seconds=float(timestamp),
            )

        if origin is None:
            raise ValueError("Video contains no decodable frames")
