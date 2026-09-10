"""Recording-directory import with real encoded footage and sidecar files."""

from dataclasses import asdict
from datetime import timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import av
import numpy as np

from src.engine.calibration import CameraCalibration
from src.engine.recording_io import load_video


class RecordingImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        with av.open(str(self.root / "video.mp4"), "w") as container:
            stream = container.add_stream("mpeg4", rate=2)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for _ in range(2):
                frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="bgr24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def metadata(self, **values):
        (self.root / "metadata.json").write_text(json.dumps(values))

    def gps(self, rows):
        (self.root / "gps.csv").write_text("timestamp,latitude,longitude,accuracy_m\n" + rows)

    def calibration(self, **overrides):
        values = asdict(CameraCalibration(64, 48, 50, 50, 32, 24, 1.5, 45, road_roi=(0, 24, 64, 48)))
        values.update(overrides)
        (self.root / "calibration.json").write_text(json.dumps(values))

    def test_full_recording_and_serialization(self):
        self.metadata(id="survey-1", route_id="route-a", recorded_at="2026-09-08T12:00:00-06:00",
                      width=64, height=48, gps_offset_seconds=.25, camera_id="0",
                      calibration_status="available", clock_mapping={"source": "recorder"})
        self.gps("2026-09-08T18:00:01Z,40.001,-105,5\n2026-09-08T18:00:00Z,40,-105,\n")
        self.calibration()
        video = load_video(self.root)
        self.assertEqual(video.id, "survey-1")
        self.assertEqual(video.route_id, "route-a")
        self.assertEqual(video.metadata.recorded_at.hour, 18)
        self.assertEqual(video.metadata.recorded_at.tzinfo, timezone.utc)
        self.assertEqual(video.metadata.gps_offset_seconds, .25)
        self.assertEqual(video.metadata.camera_id, "0")
        self.assertEqual(video.metadata.clock_mapping, {"source": "recorder"})
        self.assertEqual(video.gps[0].location.latitude, 40)
        self.assertIsNone(video.gps[0].location.accuracy_m)
        self.assertIsNotNone(video.calibration)
        self.assertTrue(video.footage_path.is_absolute())
        json.dumps(video.to_dict(), allow_nan=False)

    def test_footage_only(self):
        video = load_video(self.root)
        self.assertEqual((video.metadata.width, video.metadata.height), (64, 48))
        self.assertEqual(video.metadata.fps, 2)
        self.assertIsNone(video.metadata.recorded_at)
        self.assertEqual(video.gps, [])
        self.assertIsNone(video.calibration)
        self.assertEqual(video.metadata.calibration_status, "unavailable")

    def test_missing_and_ambiguous_video(self):
        path = self.root / "video.mp4"
        path.rename(self.root / "other.mp4")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_video(self.root)
        (self.root / "other.mp4").rename(path)
        (self.root / "video.mov").write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_video(self.root)

    def test_malformed_metadata(self):
        for content in ("{", "[]", '{"gps_offset_seconds": NaN}', '{"gps_offset_seconds": 1e999}'):
            with self.subTest(content=content):
                (self.root / "metadata.json").write_text(content)
                with self.assertRaisesRegex(ValueError, "metadata.json"):
                    load_video(self.root)
        for values in ({"recorded_at": "2026-09-08T12:00:00"}, {"width": True},
                       {"gps_offset_seconds": None}, {"fps": -1}, {"unknown": 1}):
            with self.subTest(values=values):
                self.metadata(**values)
                with self.assertRaisesRegex(ValueError, "metadata.json"):
                    load_video(self.root)

    def test_gps_errors_include_context(self):
        self.metadata(recorded_at="2026-09-08T18:00:00Z")
        for row in ("2026-09-08T18:00:00,40,-105,5\n", "2026-09-08T18:00:00Z,91,-105,5\n",
                    "2026-09-08T18:00:00Z,nan,-105,5\n", "2026-09-08T18:00:00Z,40,-105,-1\n",
                    "2026-09-08T18:00:00Z,40,-105\n"):
            with self.subTest(row=row):
                self.gps(row)
                with self.assertRaisesRegex(ValueError, "gps.csv row 2"):
                    load_video(self.root)
        (self.root / "gps.csv").write_text("time,lat,lon\n")
        with self.assertRaisesRegex(ValueError, "header"):
            load_video(self.root)

    def test_duplicate_gps_and_missing_clock(self):
        self.metadata(recorded_at="2026-09-08T18:00:00Z")
        self.gps("2026-09-08T18:00:00Z,40,-105,5\n2026-09-08T12:00:00-06:00,40,-105,5\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_video(self.root)
        self.gps("2026-09-08T18:00:00Z,40,-105,5\n")
        self.metadata()
        with self.assertRaisesRegex(ValueError, "recorded_at"):
            load_video(self.root)

    def test_calibration_geometry_and_status(self):
        self.metadata(width=128)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            load_video(self.root)
        self.metadata(calibration_status="available")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            load_video(self.root)
        self.metadata()
        self.calibration(image_width=128)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            load_video(self.root)
        self.calibration(road_roi=[0, 24.5, 64, 48])
        with self.assertRaisesRegex(ValueError, "integers"):
            load_video(self.root)
        (self.root / "calibration.json").write_text('{"fx": 50}')
        with self.assertRaisesRegex(ValueError, "calibration.json"):
            load_video(self.root)

    def test_corrupt_video_is_rejected(self):
        (self.root / "video.mp4").write_bytes(b"not a video")
        with self.assertRaises(av.error.InvalidDataError):
            load_video(self.root)


if __name__ == "__main__":
    unittest.main()
