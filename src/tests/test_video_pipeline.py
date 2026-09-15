"""Real decoding and crack analysis with deterministic model predictions."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import av
import numpy as np

from src.engine.constants import DamageType
from src.engine.calibration import CameraCalibration
from src.engine.motion import MotionEstimate
from src.engine.gps import location_at
from src.engine.video import GPSPoint, Location, Video, VideoMetadata
from src.engine.video_pipeline import VideoPipeline


class VideoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        weights = self.root / "weights.pt"
        weights.write_bytes(b"test checkpoint")
        model = Mock(task="segment", names={0: "crack", 1: "pothole"})
        self.model = model
        with patch("src.engine.segmentation.YOLO", return_value=model):
            self.pipeline = VideoPipeline(weights, img_size=(96, 96))
        masks = np.zeros((3, 96, 96), dtype=np.uint8)
        masks[0, 10:45, 20:23] = 1
        masks[1, 50:85, 20:23] = 1
        masks[2, 10:20, 70:80] = 1
        self.prediction = SimpleNamespace(
            masks=SimpleNamespace(data=masks),
            boxes=SimpleNamespace(conf=np.array([.8, .9, .95]), cls=np.array([0, 0, 1])),
        )
        model.predict.return_value = [Mock()]
        model.predict.return_value[0].cpu.return_value.numpy.return_value = self.prediction

    def make_video(self, suffix=".mp4"):
        path = self.root / f"survey{suffix}"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=2)
            stream.width, stream.height = 192, 96
            stream.pix_fmt = "yuv420p"
            for _ in range(3):
                frame = av.VideoFrame.from_ndarray(np.zeros((96, 192, 3), dtype=np.uint8), format="bgr24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        start = datetime(2026, 9, 8, tzinfo=timezone.utc)
        return Video(path, metadata=VideoMetadata(recorded_at=start), gps=[
            GPSPoint(start, Location(40, -105, 5)),
            GPSPoint(start + timedelta(seconds=1), Location(40.001, -105.001, 5)),
        ])

    def test_formats_merging_measurement_and_source_coordinates(self):
        for suffix in (".mp4", ".mov", ".avi", ".mkv"):
            with self.subTest(suffix=suffix):
                video = self.make_video(suffix)
                damages = list(self.pipeline.process(video, sample_fps=1))
                self.assertEqual(len(damages), 4)  # One merged crack and one pothole per sampled frame.
                self.assertEqual([d.frame_index for d in damages], [0, 0, 2, 2])
                for damage in damages:
                    self.assertEqual(damage.video_id, video.id)
                    self.assertEqual(damage.mask.shape, (96, 192))
                    self.assertEqual(damage.dimensions.area.value, np.count_nonzero(damage.mask))
                    self.assertTrue(np.isfinite(damage.dimensions.thickness.value))
                    self.assertGreater(damage.dimensions.thickness.value, 0)
                    self.assertIsNone(damage.severity)
                    json.dumps(damage.to_dict(), allow_nan=False)
                crack, pothole = damages[:2]
                self.assertEqual(crack.type, DamageType.CRACK)
                self.assertGreater(crack.dimensions.length.value, 60)
                self.assertLess(crack.dimensions.thickness.value, 10)
                self.assertEqual(pothole.bounding_box, (140, 10, 160, 20))
                self.assertEqual(pothole.dimensions.area.value, 200)
                self.assertEqual(damages[-1].location, video.gps[-1].location)
        self.assertTrue(self.model.predict.call_args.kwargs["retina_masks"])

    def test_gps_interpolation_and_missing_coverage(self):
        video = self.make_video()
        damages = list(self.pipeline.process(video))
        self.assertAlmostEqual(damages[2].location.latitude, 40.0005)
        self.assertIsNone(damages[2].location.accuracy_m)
        damages = list(self.pipeline.process(video, max_gps_gap_seconds=.25))
        self.assertIsNone(damages[2].location)
        video.metadata.gps_offset_seconds = 10
        self.assertTrue(all(d.location is None for d in self.pipeline.process(video)))
        video.metadata.recorded_at = None
        self.assertTrue(all(d.location is None for d in self.pipeline.process(video)))

    def test_export_masks_and_manifest(self):
        video = self.make_video()
        output = self.root / "results"
        manifest = json.loads(self.pipeline.write(video, output, sample_fps=1).read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["damage_count"], 4)
        self.assertEqual(manifest["video"]["metadata"]["width"], 192)
        observations = [json.loads(line) for line in (output / "damages.jsonl").read_text().splitlines()]
        with np.load(output / observations[0]["mask_path"]) as arrays:
            self.assertEqual(arrays["mask"].shape, (96, 192))
            self.assertEqual(arrays["skeleton"].shape, (96, 192))
        with self.assertRaises(FileExistsError):
            self.pipeline.write(video, output)

    def test_frame_results_include_empty_frames(self):
        video = self.make_video()
        self.prediction.masks = None
        frames = list(self.pipeline.process_frames(video))
        self.assertEqual([r.frame.index for r in frames], [0, 1, 2])
        self.assertTrue(all(r.damages == [] for r in frames))
        self.assertEqual(frames[0].location, video.gps[0].location)
        self.assertEqual(frames[0].frame.image.shape, (96, 192, 3))

    def test_temporal_exports_associations_and_visits_empty_frames(self):
        from src.engine.temporal import Association
        video = self.make_video()
        temporal = Mock()
        temporal.process.side_effect = lambda video, result: [
            Association(str(d.id), "existing-defect", "matched", .9) for d in result.damages]
        output = self.root / "temporal"
        manifest = json.loads(self.pipeline.write(video, output, temporal=temporal).read_text())
        rows = [json.loads(line) for line in (output / "damages.jsonl").read_text().splitlines()]
        self.assertTrue(manifest["temporal"])
        self.assertTrue(all(row["temporal"]["defect_id"] == "existing-defect" for row in rows))
        self.assertEqual(temporal.process.call_count, 3)
        temporal.reset_mock()
        self.prediction.masks = None
        self.pipeline.write(video, self.root / "empty-temporal", temporal=temporal)
        self.assertEqual(temporal.process.call_count, 3)

    def test_no_detections_and_failure_status(self):
        video = self.make_video()
        self.prediction.masks = None
        output = self.root / "empty"
        manifest = json.loads(self.pipeline.write(video, output).read_text())
        self.assertEqual(manifest["damage_count"], 0)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual((output / "damages.jsonl").read_text(), "")
        self.model.predict.side_effect = RuntimeError("inference failed")
        output = self.root / "failed"
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.pipeline.write(video, output)
        self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "failed")

    def test_calibrated_and_triangulated_measurements_export_separately(self):
        video = self.make_video()
        video.calibration = CameraCalibration(192, 96, 200, 200, 96, 48, 2, 90,
                                               road_roi=(0, 0, 192, 96))
        self.pipeline.triangulation = True
        motion = MotionEstimate("ok", "camera_height", baseline_m=.5, inlier_count=30,
                                road_normal=[0, 0, 1], road_distance_m=4,
                                support_pixels=[[0, 0], [192, 0], [192, 96], [0, 96]])
        with patch("src.engine.video_pipeline.RoadTracker") as tracker:
            tracker.return_value.update.return_value = motion
            output = self.root / "metric-results"
            self.pipeline.write(video, output)
        observations = [json.loads(line) for line in (output / "damages.jsonl").read_text().splitlines()]
        for damage in observations:
            self.assertEqual(damage["dimensions"]["area"]["unit"], "px")
            self.assertEqual(damage["metric_status"], "ok")
            self.assertEqual(damage["triangulation_status"], "ok")
            self.assertAlmostEqual(damage["triangulated_dimensions"]["area_m2"],
                                   4*damage["metric_dimensions"]["area_m2"])
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertTrue(manifest["triangulation"])
        self.assertEqual(manifest["video"]["calibration"]["camera_height_m"], 2)

    def test_unavailable_triangulation_preserves_calibrated_measurements(self):
        video = self.make_video()
        video.calibration = CameraCalibration(192, 96, 200, 200, 96, 48, 2, 90,
                                               road_roi=(0, 0, 192, 96))
        self.pipeline.triangulation = True
        damages = list(self.pipeline.process(video))  # Blank video has no trackable features.
        self.assertEqual(damages[0].triangulation_status, "first_frame")
        self.assertEqual(damages[-1].triangulation_status, "insufficient_features")
        self.assertTrue(all(d.metric_dimensions is not None and d.triangulated_dimensions is None for d in damages))
        video.calibration = None
        with self.assertRaisesRegex(ValueError, "calibration"):
            list(self.pipeline.process(video))


if __name__ == "__main__":
    unittest.main()
