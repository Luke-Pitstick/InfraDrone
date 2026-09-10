"""Exercise decoding and sampling against a real variable-timing video."""

from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import av
import numpy as np

from src.engine.video import Video
from src.engine.video_io import read_frames


class VideoReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "survey.mkv"
        with av.open(str(self.path), "w") as container:
            stream = container.add_stream("ffv1", rate=10)
            stream.width, stream.height = 32, 24
            stream.pix_fmt = "bgr0"
            stream.time_base = Fraction(1, 1000)
            stream.codec_context.time_base = Fraction(1, 1000)
            for timestamp in (2000, 2100, 2400, 3000, 3100):
                pixels = np.zeros((24, 32, 3), dtype=np.uint8)
                pixels[:, :, 2] = 255
                frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
                frame.pts = timestamp
                frame.time_base = Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def test_preserves_variable_timing_and_bgr_pixels(self):
        video = Video(self.path)
        frames = list(read_frames(video))
        self.assertEqual([f.index for f in frames], list(range(5)))
        np.testing.assert_allclose([f.timestamp_seconds for f in frames], [0, .1, .4, 1, 1.1])
        self.assertEqual(frames[0].image.shape, (24, 32, 3))
        np.testing.assert_array_equal(frames[0].image[0, 0], [0, 0, 255])
        self.assertEqual((video.metadata.width, video.metadata.height), (32, 24))

    def test_sampling_keeps_source_indices_without_duplicates(self):
        frames = list(read_frames(Video(self.path), sample_fps=2))
        self.assertEqual([f.index for f in frames], [0, 3])
        self.assertEqual([f.timestamp_seconds for f in frames], [0, 1])
        self.assertEqual(len(list(read_frames(Video(self.path), sample_fps=100))), 5)

    def test_invalid_sampling_rates(self):
        for rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                next(read_frames(Video(self.path), sample_fps=rate))

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            next(read_frames(Video(self.path.with_name("missing.mp4"))))


if __name__ == "__main__":
    unittest.main()
