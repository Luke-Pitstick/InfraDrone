"""Metric road-plane geometry for a fixed, calibrated camera."""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class MetricDimensions:
    """Surface area, total skeleton length, and area/length mean width in meters.

    These are estimates on a locally flat road, not pothole depth measurements.
    """

    length_m: float
    mean_width_m: float | None
    area_m2: float


@dataclass
class CameraCalibration:
    """Intrinsics must describe the encoded video pixels after crop/zoom.

    Camera axes: x right, y down, z forward. pitch_degrees is downward tilt
    from the road plane (0 = horizontal, 90 = looking straight down).
    roll_degrees rotates the road normal clockwise in the image. Height is
    perpendicular distance to the road, not GPS altitude. Distortion is OpenCV
    (k1, k2, p1, p2, k3); use zero only for already corrected/calibrated images.
    road_roi is an optional (left, top, right, bottom) road-only pixel rectangle,
    required when tracking motion. Stabilization and changing crops invalidate
    fixed intrinsics unless corrected before this pipeline.
    """

    image_width: int
    image_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_height_m: float
    pitch_degrees: float
    roll_degrees: float = 0.0
    distortion: tuple[float, float, float, float, float] = (0, 0, 0, 0, 0)
    road_roi: tuple[int, int, int, int] | None = None
    max_range_m: float = 30.0

    def __post_init__(self):
        values = (self.fx, self.fy, self.cx, self.cy, self.camera_height_m,
                  self.pitch_degrees, self.roll_degrees, self.max_range_m, *self.distortion)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Calibration values must be finite")
        if min(self.image_width, self.image_height, self.fx, self.fy, self.camera_height_m) <= 0:
            raise ValueError("Image dimensions, focal lengths, and camera height must be positive")
        if not 0 <= self.pitch_degrees <= 90 or self.max_range_m <= self.camera_height_m:
            raise ValueError("Pitch must be 0–90 degrees and range must exceed camera height")
        if len(self.distortion) != 5:
            raise ValueError("Expected five OpenCV distortion coefficients")
        if self.road_roi is not None:
            left, top, right, bottom = self.road_roi
            if not (0 <= left < right <= self.image_width and 0 <= top < bottom <= self.image_height):
                raise ValueError("road_roi must lie inside the calibrated image")

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)

    @property
    def road_normal(self) -> np.ndarray:
        pitch, roll = np.radians([self.pitch_degrees, self.roll_degrees])
        return np.array([-np.sin(roll) * np.cos(pitch), np.cos(roll) * np.cos(pitch), np.sin(pitch)])

    def check_shape(self, shape: tuple) -> None:
        if shape[:2] != (self.image_height, self.image_width):
            raise ValueError("Frame dimensions do not match calibration; calibrate the encoded video resolution")

    def normalized_points(self, pixels: np.ndarray) -> np.ndarray:
        return cv2.undistortPoints(
            np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2),
            self.matrix, np.asarray(self.distortion, dtype=float),
        ).reshape(-1, 2)

    def project(self, pixels: np.ndarray, normal: np.ndarray | None = None,
                distance_m: float | None = None) -> np.ndarray | None:
        """Intersect pixel rays with n·X=d; reject points behind/too far from camera."""
        normal = self.road_normal if normal is None else normal
        distance_m = self.camera_height_m if distance_m is None else distance_m
        rays = np.column_stack([self.normalized_points(pixels), np.ones(len(pixels))])
        denominator = rays @ normal
        if np.any(denominator <= 1e-6) or distance_m <= 0:
            return None
        points = rays * (distance_m / denominator[:, None])
        if not np.isfinite(points).all() or np.any(np.linalg.norm(points, axis=1) > self.max_range_m):
            return None
        return points

    def measure(self, mask: np.ndarray, normal: np.ndarray | None = None,
                distance_m: float | None = None) -> MetricDimensions | None:
        """Project mask pixel cells and skeleton edges, without a single global scale.

        Reject the entire measurement if any foreground cell leaves the valid
        road/range region. Work in chunks so projection memory stays bounded.
        """
        self.check_shape(mask.shape)
        mask = mask > 0
        rows, columns = np.nonzero(mask)
        if not len(rows):
            return None
        if self.road_roi is not None:
            left, top, right, bottom = self.road_roi
            if columns.min() < left or columns.max() >= right or rows.min() < top or rows.max() >= bottom:
                return None
        area = 0.0
        offsets = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
        for start in range(0, len(rows), 8192):
            centers = np.column_stack([columns[start:start+8192], rows[start:start+8192]])
            cells = self.project((centers[:, None, :] + offsets).reshape(-1, 2), normal, distance_m)
            if cells is None:
                return None
            cells = cells.reshape(-1, 4, 3)
            for a, b, c in ((0, 1, 2), (0, 2, 3)):
                area += float(np.linalg.norm(np.cross(cells[:, b]-cells[:, a], cells[:, c]-cells[:, a]), axis=1).sum() / 2)
        skeleton = skeletonize(mask)
        rows, columns = np.nonzero(skeleton)
        pixels = np.column_stack([columns, rows])
        points = self.project(pixels, normal, distance_m)
        if points is None:
            return None
        lookup = {(int(row), int(col)): i for i, (row, col) in enumerate(zip(rows, columns))}
        length = 0.0
        for (row, col), index in lookup.items():
            for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
                other = lookup.get((row + dr, col + dc))
                if other is not None:
                    # Avoid an extra diagonal across a right-angle connection.
                    if dr and dc and ((row, col + dc) in lookup or (row + dr, col) in lookup):
                        continue
                    length += float(np.linalg.norm(points[index] - points[other]))
        return MetricDimensions(length, area / length if length > 0 else None, area)
