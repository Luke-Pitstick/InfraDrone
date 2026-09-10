"""Exercise real ORB registration, persistence, and conservative association."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from src.engine.calibration import CameraCalibration
from src.engine.constants import DamageType
from src.engine.models import Damage
from src.engine.motion import extract_features, match_features
from src.engine.temporal import TemporalEngine
from src.engine.temporal_store import TemporalStore
from src.engine.video import Location, Video, VideoFrame
from src.engine.video_pipeline import FrameResult


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'temporal.db'
        self.store = TemporalStore(self.path)
        self.addCleanup(lambda: self.store.close())
        self.engine = TemporalEngine(self.store, extractor=extract_features, matcher=match_features)
        self.calibration = CameraCalibration(400, 300, 350, 350, 200, 150, 2, 90,
                                             road_roi=(0, 0, 400, 300))
        self.image = np.random.default_rng(4).integers(0, 256, (300, 400, 3), dtype=np.uint8)
        self.video = Video(Path('survey.mp4'), route_id='road', calibration=self.calibration)

    def result(self, index=0, shift=0, positions=(170,), image=None):
        damages = []
        for x in positions:
            mask = np.zeros((300, 400), np.uint8)
            mask[130:170, x+shift:x+shift+3] = 1
            damages.append(Damage(DamageType.CRACK, .9, mask=mask, skeleton=mask.copy()))
        if image is None:
            image = cv2.warpAffine(self.image, np.float32([[1, 0, shift], [0, 1, 0]]), (400, 300))
        return FrameResult(VideoFrame(image, index, float(index)), Location(40, -105, 5), damages)

    def test_default_uses_learned_features_once(self):
        from unittest.mock import patch
        with patch('src.engine.learned_features.LearnedFeatures') as factory:
            factory.return_value.extract.side_effect = extract_features
            factory.return_value.match.side_effect = match_features
            engine = TemporalEngine(self.store, device='cpu')
            first = engine.process(self.video, self.result())[0]
            second = engine.process(self.video, self.result(1))[0]
            factory.assert_called_once_with('cpu')
            self.assertEqual(second.defect_id, first.defect_id)
            self.assertGreater(factory.return_value.match.call_count, 0)

    def test_shifted_frame_and_second_survey_reuse_id_after_reopen(self):
        first = self.engine.process(self.video, self.result())[0]
        self.assertEqual(first.status, 'baseline')
        shifted = self.engine.process(self.video, self.result(1, 8))[0]
        self.assertEqual(shifted.status, 'matched')
        self.assertEqual(first.defect_id, shifted.defect_id)
        self.store.close()
        self.store = TemporalStore(self.path)
        self.engine = TemporalEngine(self.store, extractor=extract_features, matcher=match_features)
        second_video = Video(Path('later.mp4'), route_id='road', calibration=self.calibration)
        later = self.engine.process(second_video, self.result(0, 4))[0]
        self.assertEqual(later.defect_id, first.defect_id)
        self.engine.process(second_video, self.result(0, 4))
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM observations').fetchone()[0], 3)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM defects').fetchone()[0], 1)

    def test_different_resolution_and_intrinsics(self):
        first = self.engine.process(self.video, self.result())[0]
        calibration = CameraCalibration(800, 600, 700, 700, 400, 300, 2, 90,
                                        road_roi=(0, 0, 800, 600))
        video = Video(Path('larger.mp4'), route_id='road', calibration=calibration)
        result = self.result()
        result.frame.image = cv2.resize(self.image, (800, 600))
        for damage in result.damages:
            damage.mask = cv2.resize(damage.mask, (800, 600), interpolation=cv2.INTER_NEAREST)
            damage.skeleton = damage.mask.copy()
        association = self.engine.process(video, result)[0]
        self.assertEqual(association.status, 'matched')
        self.assertEqual(association.defect_id, first.defect_id)

    def test_failed_write_rolls_back_frame_and_defect(self):
        result = self.result()
        result.damages[0].confidence = float('nan')
        with self.assertRaises(ValueError):
            self.engine.process(self.video, result)
        self.assertFalse(self.store.has_frame(self.video.id, 0))
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM defects').fetchone()[0], 0)

    def test_edge_crossing_crack_matches_on_shared_coverage(self):
        first = self.result()
        first.damages[0].mask[:] = 0
        first.damages[0].mask[:, 170:173] = 1
        first.damages[0].skeleton = first.damages[0].mask.copy()
        identity = self.engine.process(self.video, first)[0].defect_id
        second = self.result(1)
        second.damages[0].mask = first.damages[0].mask.copy()
        second.damages[0].skeleton = second.damages[0].mask.copy()
        association = self.engine.process(self.video, second)[0]
        self.assertEqual(association.status, 'matched')
        self.assertEqual(association.defect_id, identity)

    def test_tiny_shared_sliver_is_unresolved(self):
        first = self.result()
        first.damages[0].mask[:] = 0
        first.damages[0].mask[:, :35] = 1
        first.damages[0].skeleton = first.damages[0].mask.copy()
        self.engine.process(self.video, first)
        second = self.result(1)
        second.damages[0].mask = first.damages[0].mask.copy()
        second.damages[0].skeleton = second.damages[0].mask.copy()
        self.assertEqual(self.engine.process(self.video, second)[0].status, 'unresolved')

    def test_distinct_crack_with_aligned_pavement_gets_new_id(self):
        original = self.engine.process(self.video, self.result())[0]
        other = self.engine.process(self.video, self.result(1, positions=(220,)))[0]
        self.assertEqual(other.status, 'new')
        self.assertNotEqual(other.defect_id, original.defect_id)

    def test_unmatched_partial_coverage_does_not_create_new_id(self):
        self.engine.process(self.video, self.result(positions=(220,)))
        second = self.result(1)
        second.damages[0].mask[:, 170:173] = 1
        second.damages[0].skeleton = second.damages[0].mask.copy()
        association = self.engine.process(self.video, second)[0]
        self.assertEqual(association.status, 'unresolved')
        self.assertIsNone(association.defect_id)

    def test_tiny_fragment_is_insufficient_even_when_fully_supported(self):
        first = self.result()
        first.damages[0].mask[:] = 0
        first.damages[0].mask[145:149, 170:174] = 1
        first.damages[0].skeleton = first.damages[0].mask.copy()
        self.engine.process(self.video, first)
        second = self.result(1)
        second.damages[0].mask = first.damages[0].mask.copy()
        second.damages[0].skeleton = second.damages[0].mask.copy()
        self.assertEqual(self.engine.process(self.video, second)[0].status, 'unresolved')

    def test_benchmark_empty_mask_is_not_a_damage_observation(self):
        from src.tests.benchmarks.temporal_dataset import observation
        result = observation(self.image, np.zeros(self.image.shape[:2], np.uint8), Location(40, -105, 5))
        self.assertEqual(result.damages, [])
        self.assertEqual(self.engine.process(self.video, result), [])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM defects').fetchone()[0], 0)

    def test_failed_alignment_is_unresolved(self):
        self.engine.process(self.video, self.result())
        result = self.engine.process(self.video, self.result(1, image=np.zeros_like(self.image)))[0]
        self.assertEqual(result.status, 'unresolved')
        self.assertIsNone(result.defect_id)

    def test_empty_reference_and_new_damage(self):
        self.assertEqual(self.engine.process(self.video, self.result(positions=())), [])
        self.assertTrue(self.store.has_frame(self.video.id, 0))
        result = self.engine.process(self.video, self.result(1))[0]
        self.assertEqual(result.status, 'new')

    def test_competing_detections_and_outside_support_are_unresolved(self):
        self.engine.process(self.video, self.result())
        results = self.engine.process(self.video, self.result(1, positions=(170, 171, 3)))
        self.assertTrue(all(r.status == 'unresolved' for r in results))

    def test_measurements_and_masks_are_preserved(self):
        frame = self.result()
        self.engine.process(self.video, frame)
        row = self.store.observations(self.video.id, 0)[0]
        from src.engine.temporal_store import unpack
        np.testing.assert_array_equal(unpack(row['masks'])['mask'], frame.damages[0].mask)
        self.assertIn('metric_dimensions', row['payload'])


if __name__ == '__main__':
    unittest.main()
