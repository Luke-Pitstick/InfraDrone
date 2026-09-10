"""Known-scale synthetic camera scenes, not camera-specific accuracy claims."""

from dataclasses import asdict
import json
import unittest

import cv2
import numpy as np

from src.engine.calibration import CameraCalibration
from src.engine.motion import RoadTracker, gps_baseline, reconstruct_road
from src.engine.video import Location, VideoFrame


class CalibrationTests(unittest.TestCase):
    def test_downward_camera_known_area_and_scale(self):
        camera = CameraCalibration(200, 200, 100, 100, 100, 100, 2, 90)
        np.testing.assert_allclose(camera.project(np.array([[100, 100], [110, 100]])),
                                   [[0, 0, 2], [.2, 0, 2]], atol=1e-10)
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[80:100, 80:120] = 1
        result = camera.measure(mask)
        self.assertAlmostEqual(result.area_m2, .32, places=8)
        self.assertGreater(result.length_m, 0)
        self.assertAlmostEqual(result.mean_width_m, result.area_m2/result.length_m)
        camera.camera_height_m = 4
        doubled = camera.measure(mask)
        self.assertAlmostEqual(doubled.area_m2/result.area_m2, 4)
        self.assertAlmostEqual(doubled.length_m/result.length_m, 2)
        json.dumps(asdict(doubled), allow_nan=False)

    def test_perspective_distortion_and_rejection(self):
        camera = CameraCalibration(640, 480, 500, 500, 320, 240, 1.5, 0,
                                   distortion=(.1, -.01, .001, 0, 0), max_range_m=20)
        points = np.array([[0, 1.5, 5], [1, 1.5, 10]], dtype=float)
        pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera.matrix, np.array(camera.distortion))
        np.testing.assert_allclose(camera.project(pixels.reshape(-1, 2)), points, atol=1e-5)
        self.assertIsNone(camera.project(np.array([[320, 240]])))  # Horizon.
        self.assertIsNone(camera.project(np.array([[320, 100]])))  # Above road.
        self.assertIsNone(camera.project(np.array([[320, 241]])))  # Beyond range.
        with self.assertRaises(ValueError):
            camera.measure(np.zeros((240, 320), dtype=np.uint8))
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[239:260, 300:320] = 1
        self.assertIsNone(camera.measure(mask))  # No partial measurement across horizon.

    def test_roi_and_invalid_configuration(self):
        camera = CameraCalibration(200, 200, 100, 100, 100, 100, 2, 90, road_roi=(50, 50, 150, 150))
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[40:60, 60:80] = 1
        self.assertIsNone(camera.measure(mask))
        with self.assertRaises(ValueError):
            CameraCalibration(200, 200, 0, 100, 100, 100, 2, 90)
        with self.assertRaises(ValueError):
            CameraCalibration(200, 200, 100, 100, 100, 100, -2, 90)


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        cv2.setRNGSeed(7)
        self.camera = CameraCalibration(640, 480, 500, 500, 320, 240, 1.5, 35,
                                        road_roi=(40, 200, 600, 475))
        x, y = np.meshgrid(np.linspace(60, 580, 12), np.linspace(250, 440, 8))
        self.first = np.column_stack([x.ravel(), y.ravel()])
        points = self.camera.project(self.first)
        pitch = np.radians(35)
        self.translation = np.array([.1, .8*np.sin(pitch), -.8*np.cos(pitch)])
        current = points+self.translation
        self.second = current[:, :2]/current[:, 2:]*[500, 500]+[320, 240]

    def test_reconstructs_known_metric_motion_and_plane(self):
        result = reconstruct_road(self.first, self.second, self.camera)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.baseline_m, np.linalg.norm(self.translation), places=5)
        self.assertAlmostEqual(result.road_distance_m, 1.5, places=5)
        self.assertLess(result.median_reprojection_error_px, .01)
        np.testing.assert_allclose(result.road_normal, self.camera.road_normal, atol=1e-5)
        json.dumps(asdict(result), allow_nan=False)

    def test_gps_scale_is_explicit_and_not_replaced_by_height(self):
        result = reconstruct_road(self.first, self.second, self.camera, "gps", 2*np.linalg.norm(self.translation))
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.road_distance_m, 3, places=5)
        rejected = reconstruct_road(self.first, self.second, self.camera, "gps")
        self.assertEqual(rejected.status, "insufficient_gps_baseline")
        self.assertIsNone(rejected.road_distance_m)
        self.assertIsNone(gps_baseline(Location(40, -105, 5), Location(40.00001, -105, 5)))
        self.assertIsNone(gps_baseline(Location(40, -105), Location(40.001, -105, 5)))
        self.assertGreater(gps_baseline(Location(40, -105, 1), Location(40.001, -105, 1)), 100)

    def test_outliers_and_stationary_camera(self):
        corrupt = self.second.copy()
        corrupt[:10] = np.random.default_rng(7).uniform(0, 500, (10, 2))
        result = reconstruct_road(self.first, corrupt, self.camera)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.baseline_m, np.linalg.norm(self.translation), places=4)
        stationary = reconstruct_road(self.first, self.first, self.camera)
        self.assertNotEqual(stationary.status, "ok")
        self.assertIsNone(stationary.baseline_m)
        rotation, _ = cv2.Rodrigues(np.array([0, .05, 0]))
        rays = np.column_stack([self.camera.normalized_points(self.first), np.ones(len(self.first))]) @ rotation.T
        rotated = rays[:, :2]/rays[:, 2:]*[500, 500]+[320, 240]
        self.assertNotEqual(reconstruct_road(self.first, rotated, self.camera).status, "ok")

    def test_tracker_feature_matching_on_synthetic_road(self):
        # Texture rendered through the known plane homography exercises ORB too.
        pixels = np.random.default_rng(2).integers(0, 256, (480, 640), dtype=np.uint8)
        image = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        translation = self.translation * .2
        homography = self.camera.matrix @ (np.eye(3)+np.outer(translation, self.camera.road_normal)/1.5) @ np.linalg.inv(self.camera.matrix)
        next_image = cv2.warpPerspective(image, homography, (640, 480))
        tracker = RoadTracker(self.camera, "camera_height")
        self.assertEqual(tracker.update(VideoFrame(image, 0, 0), None).status, "first_frame")
        estimate = tracker.update(VideoFrame(next_image, 1, .1), None)
        self.assertEqual(estimate.status, "ok")
        self.assertAlmostEqual(estimate.baseline_m, np.linalg.norm(translation), delta=.05)
        self.assertEqual(estimate.reference_frame_index, 0)


if __name__ == "__main__":
    unittest.main()
