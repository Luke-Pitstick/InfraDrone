"""Two-view road reconstruction using OpenCV and an explicit metric scale.

The road is approximately planar, so use homography decomposition rather than
an unconstrained essential-matrix solve on a degenerate planar scene. Matched
points are then triangulated and checked for parallax and reprojection error.
Camera-height scaling shares method one's height assumption; it is not an
independent validation of that height. GPS scaling is available separately.
"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .calibration import CameraCalibration
from .video import Location, VideoFrame


@dataclass
class MotionEstimate:
    status: str
    scale_source: str
    reference_frame_index: int | None = None
    baseline_m: float | None = None
    inlier_count: int = 0
    median_parallax_degrees: float | None = None
    median_reprojection_error_px: float | None = None
    road_normal: list[float] | None = None
    road_distance_m: float | None = None
    support_pixels: list[list[float]] | None = None

    def covers(self, bounding_box: tuple) -> bool:
        """Require the entire box inside the matched feature support region."""
        if self.status != "ok":
            return False
        x1, y1, x2, y2 = bounding_box
        hull = np.asarray(self.support_pixels, dtype=np.float32)
        return all(cv2.pointPolygonTest(hull, point, False) >= 0
                   for point in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))


@dataclass
class FrameFeatures:
    points: np.ndarray
    descriptors: np.ndarray
    image_size: np.ndarray


def extract_features(image: np.ndarray, calibration: CameraCalibration) -> FrameFeatures:
    """Extract road-only ORB features in original pixel coordinates."""
    calibration.check_shape(image.shape)
    if calibration.road_roi is None:
        raise ValueError("Frame alignment requires a road-only road_roi")
    roi = np.zeros(image.shape[:2], dtype=np.uint8)
    left, top, right, bottom = calibration.road_roi
    roi[top:bottom, left:right] = 255
    keys, descriptors = cv2.ORB_create(nfeatures=1500).detectAndCompute(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), roi)
    return FrameFeatures(np.array([k.pt for k in keys], dtype=float).reshape(-1, 2),
                         descriptors if descriptors is not None else np.empty((0, 32), dtype=np.uint8),
                         np.array([image.shape[1], image.shape[0]], dtype=np.float32))


def match_features(first: FrameFeatures, second: FrameFeatures) -> tuple[np.ndarray, np.ndarray]:
    if len(first.descriptors) == 0 or len(second.descriptors) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(first.descriptors, second.descriptors, k=2)
    matches = sorted((p[0] for p in pairs if len(p) == 2 and p[0].distance < .75*p[1].distance),
                     key=lambda m: m.distance)
    used, unique = set(), []
    for match in matches:
        if match.trainIdx not in used:
            unique.append(match)
            used.add(match.trainIdx)
    return (first.points[[m.queryIdx for m in unique]], second.points[[m.trainIdx for m in unique]])


def fit_homography(first: np.ndarray, second: np.ndarray, threshold: float):
    """Fit first-to-second coordinates; reject weak or inconsistent alignment."""
    if len(first) < 20 or len(first) != len(second):
        return None, None
    transform, mask = cv2.findHomography(first, second, cv2.RANSAC, threshold)
    if transform is None or mask is None or not np.isfinite(transform).all():
        return None, None
    keep = mask.ravel().astype(bool)
    if keep.sum() < 20 or keep.mean() < .6:
        return None, None
    return transform, keep


def gps_baseline(first: Location | None, second: Location | None) -> float | None:
    """Accept only displacement exceeding three times the summed reported accuracy.

    This is a conservative heuristic, not a statistical confidence guarantee.
    Unknown accuracy (including interpolated fixes) cannot establish scale.
    """
    if first is None or second is None or first.accuracy_m is None or second.accuracy_m is None:
        return None
    values = [first.latitude, first.longitude, second.latitude, second.longitude,
              first.accuracy_m, second.accuracy_m]
    if not np.isfinite(values).all() or min(first.accuracy_m, second.accuracy_m) <= 0:
        return None
    lat1, lat2 = np.radians([first.latitude, second.latitude])
    dlat = lat2-lat1
    dlon = np.radians(second.longitude-first.longitude)
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    distance = float(6371008.8 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))
    return distance if distance > max(.1, 3*(first.accuracy_m+second.accuracy_m)) else None


def reconstruct_road(first: np.ndarray, second: np.ndarray, calibration: CameraCalibration,
                     scale_source: str = "camera_height", baseline_m: float | None = None) -> MotionEstimate:
    """Reconstruct a road plane from matched, original-resolution pixel points."""
    estimate = MotionEstimate("insufficient_matches", scale_source)
    if scale_source not in ("camera_height", "gps"):
        raise ValueError("scale_source must be camera_height or gps")
    if len(first) < 20 or len(first) != len(second):
        return estimate
    if scale_source == "gps" and (baseline_m is None or not np.isfinite(baseline_m) or baseline_m <= 0):
        estimate.status = "insufficient_gps_baseline"
        return estimate
    p1, p2 = calibration.normalized_points(first), calibration.normalized_points(second)
    focal = max(calibration.fx, calibration.fy)
    homography, keep = fit_homography(p1, p2, 2/focal)
    if homography is None:
        estimate.status = "inconsistent_matches"
        return estimate
    p1, p2, second = p1[keep], p2[keep], second[keep]
    _, rotations, translations, normals = cv2.decomposeHomographyMat(homography, np.eye(3))
    candidates = []
    for rotation, translation, normal in zip(rotations, translations, normals):
        alignment = float(normal.ravel() @ calibration.road_normal)
        if alignment >= np.cos(np.radians(20)):
            candidates.append((alignment, rotation, translation.ravel(), normal.ravel()))
    if not candidates:
        estimate.status = "road_pose_mismatch"
        return estimate
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0]-candidates[1][0] < .01:
        estimate.status = "ambiguous_pose"
        return estimate
    _, rotation, translation, normal = candidates[0]
    magnitude = float(np.linalg.norm(translation))
    if magnitude < 1e-5:
        estimate.status = "insufficient_translation"
        return estimate
    distance = calibration.camera_height_m if scale_source == "camera_height" else baseline_m/magnitude
    translation = translation * distance
    projection1 = np.column_stack([np.eye(3), np.zeros(3)])
    projection2 = np.column_stack([rotation, translation])
    homogeneous = cv2.triangulatePoints(projection1, projection2, p1.T, p2.T)
    finite = np.abs(homogeneous[3]) > 1e-10
    points = np.full((len(p1), 3), np.nan)
    points[finite] = (homogeneous[:3, finite] / homogeneous[3, finite]).T
    current = points @ rotation.T + translation
    rays1 = np.column_stack([p1, np.ones(len(p1))])
    rays2 = np.column_stack([p2, np.ones(len(p2))]) @ rotation
    cosine = np.sum(rays1*rays2, axis=1)/(np.linalg.norm(rays1, axis=1)*np.linalg.norm(rays2, axis=1))
    parallax = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        error = np.maximum(np.linalg.norm(points[:, :2]/points[:, 2:]-p1, axis=1),
                           np.linalg.norm(current[:, :2]/current[:, 2:]-p2, axis=1))*focal
    valid = (finite & np.isfinite(error) & (error <= 2) & (parallax >= .5)
             & (points[:, 2] > 0) & (current[:, 2] > 0)
             & (np.linalg.norm(current, axis=1) <= calibration.max_range_m))
    if valid.sum() < 20 or valid.mean() < .6:
        estimate.status = "insufficient_parallax_or_geometry"
        return estimate
    current_normal = rotation @ normal
    current_distance = float(distance + current_normal @ translation)
    if current_distance <= 0:
        estimate.status = "invalid_road_plane"
        return estimate
    estimate.status = "ok"
    estimate.baseline_m = float(np.linalg.norm(translation))
    estimate.inlier_count = int(valid.sum())
    estimate.median_parallax_degrees = float(np.median(parallax[valid]))
    estimate.median_reprojection_error_px = float(np.median(error[valid]))
    estimate.road_normal = current_normal.tolist()
    estimate.road_distance_m = current_distance
    estimate.support_pixels = cv2.convexHull(second[valid].astype(np.float32)).reshape(-1, 2).tolist()
    return estimate


class RoadTracker:
    """Bounded two-frame ORB tracking restricted to a user-specified road ROI."""

    def __init__(self, calibration: CameraCalibration, scale_source: Literal["camera_height", "gps"]):
        if calibration.road_roi is None:
            raise ValueError("Triangulation requires a road-only road_roi in the calibration")
        if scale_source not in ("camera_height", "gps"):
            raise ValueError("scale_source must be camera_height or gps")
        self.calibration = calibration
        self.scale_source = scale_source
        self.previous = None

    def update(self, frame: VideoFrame, location: Location | None) -> MotionEstimate:
        features = extract_features(frame.image, self.calibration)
        estimate = MotionEstimate("first_frame", self.scale_source)
        if self.previous is not None:
            old_index, old_features, old_location = self.previous
            first, second = match_features(old_features, features)
            estimate = reconstruct_road(first, second, self.calibration, self.scale_source,
                                        gps_baseline(old_location, location))
            if len(old_features.descriptors) == 0 or len(features.descriptors) < 2:
                estimate.status = "insufficient_features"
            estimate.reference_frame_index = old_index
        self.previous = (frame.index, features, location)
        return estimate
