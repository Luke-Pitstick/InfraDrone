"""Associate road damage in reference images; GPS only narrows the search."""

from dataclasses import dataclass
import json
from uuid import uuid4

import cv2
import numpy as np

from .calibration import CameraCalibration
from .motion import FrameFeatures, fit_homography
from .temporal_store import TemporalStore, unpack


@dataclass
class Association:
    observation_id: str
    defect_id: str | None
    status: str
    score: float | None = None


def undistort_mask(mask, calibration):
    height, width = mask.shape
    x, y = cv2.initUndistortRectifyMap(calibration.matrix, np.asarray(calibration.distortion),
                                     None, calibration.matrix, (width, height), cv2.CV_32FC1)
    return cv2.remap(mask.astype(np.uint8), x, y, cv2.INTER_NEAREST)


def overlap(first, second, tolerance=3):
    """Symmetric tolerant overlap; partial slivers cannot match a whole defect."""
    first, second = first > 0, second > 0
    if not first.any() or not second.any():
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance+1, 2*tolerance+1))
    expanded_first = cv2.dilate(first.astype(np.uint8), kernel) > 0
    expanded_second = cv2.dilate(second.astype(np.uint8), kernel) > 0
    return float(min((first & expanded_second).sum()/first.sum(),
                     (second & expanded_first).sum()/second.sum()))


def shared_evidence(mask, support):
    """Require 25% of the original mask and 64 pixels inside supported coverage.

    These conservative, pixel-based identity thresholds need field validation.
    Coverage establishes geometric support, not absence of vehicles or shadows.
    """
    visible = (mask > 0) & (support > 0)
    return visible.sum() >= 64 and visible.sum() >= .25*np.count_nonzero(mask)


class TemporalEngine:
    """Consume VideoPipeline.process_frames results with a stable route_id.

    Identity is compared only within the intersection of both inlier feature hulls.
    Insufficient shared damage, ambiguous matches, and failed registration stay unresolved.
    A new spatial neighborhood establishes baseline IDs rather than claiming damage
    appeared since a previous survey. No repair or growth inference is performed.
    """

    def __init__(self, store: TemporalStore, *, device=None, extractor=None, matcher=None):
        """Use pretrained LightGlue by default; paired overrides support comparisons."""
        if (extractor is None) != (matcher is None):
            raise ValueError('Supply both extractor and matcher, or neither')
        if extractor is None:
            from .learned_features import LearnedFeatures
            features = LearnedFeatures(device)
            extractor, matcher = features.extract, features.match
        self.store = store
        self.extractor = extractor
        self.matcher = matcher

    def process(self, video, result) -> list[Association]:
        if not video.route_id or video.calibration is None:
            raise ValueError('Temporal matching requires route_id and calibration')
        if self.store.has_frame(video.id, result.frame.index):
            return [Association(r['id'], r['defect_id'], r['status'], r['score'])
                    for r in self.store.observations(video.id, result.frame.index)]
        for damage in result.damages:
            if damage.mask is None or damage.skeleton is None:
                raise ValueError('Temporal matching requires original masks and skeletons')
        features = self.extractor(result.frame.image, video.calibration)
        references = self.store.candidates(video, result.location)
        scores = [dict() for _ in result.damages]
        covered = [False] * len(result.damages)
        blocked = [False] * len(result.damages)
        for reference in references:
            calibration = CameraCalibration(**json.loads(reference['calibration']))
            saved = FrameFeatures(**unpack(reference['features']))
            first, second = self.matcher(features, saved)
            first = video.calibration.normalized_points(first) if len(first) else first
            second = calibration.normalized_points(second) if len(second) else second
            transform, keep = fit_homography(first, second, 2/max(calibration.fx, calibration.fy))
            if transform is None:
                continue
            # Homography operates on undistorted coordinates, with each camera's K.
            warp = calibration.matrix @ transform @ np.linalg.inv(video.calibration.matrix)
            height, width = calibration.image_height, calibration.image_width
            pixels = second[keep] * [calibration.fx, calibration.fy] + [calibration.cx, calibration.cy]
            hull = cv2.convexHull(pixels.astype(np.float32))
            support = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(support, hull.astype(np.int32), 1)
            source_pixels = first[keep]*[video.calibration.fx, video.calibration.fy] + [video.calibration.cx, video.calibration.cy]
            source_support = np.zeros(result.frame.image.shape[:2], np.uint8)
            cv2.fillConvexPoly(source_support, cv2.convexHull(source_pixels.astype(np.float32)).astype(np.int32), 1)
            # Intersection of both inlier hulls, represented in reference pixels.
            support &= cv2.warpPerspective(source_support, warp, (width, height), flags=cv2.INTER_NEAREST)
            source_support = cv2.warpPerspective(support, np.linalg.inv(warp),
                (video.calibration.image_width, video.calibration.image_height), flags=cv2.INTER_NEAREST)
            previous = self.store.observations(reference['video_id'], reference['frame_index'])
            previous = [(r, unpack(r['masks']), json.loads(r['payload'])) for r in previous]
            for index, damage in enumerate(result.damages):
                source = undistort_mask(damage.mask, video.calibration)
                if not shared_evidence(source, source_support):
                    continue
                mask = cv2.warpPerspective(source, warp, (width, height), flags=cv2.INTER_NEAREST) * support
                # Partial coverage can establish identity, but cannot establish novelty.
                covered[index] |= not np.any((source > 0) & (source_support == 0))
                skeleton = cv2.warpPerspective(undistort_mask(damage.skeleton, video.calibration),
                                               warp, (width, height), flags=cv2.INTER_NEAREST) * support
                for row, arrays, payload in previous:
                    old_mask = undistort_mask(arrays['mask'], calibration)
                    old_skeleton = undistort_mask(arrays['skeleton'], calibration)
                    mask_score = overlap(mask, old_mask * support)
                    skeleton_score = overlap(skeleton, old_skeleton * support)
                    score = max(mask_score, skeleton_score)
                    if score >= .25 and not shared_evidence(old_mask, support):
                        blocked[index] = True
                        continue
                    if score < .25:
                        continue
                    if score < .6 or payload['type'] != damage.type.value or row['defect_id'] is None:
                        blocked[index] = True
                    else:
                        identity = row['defect_id']
                        scores[index][identity] = max(score, scores[index].get(identity, 0))
        associations = []
        for index, damage in enumerate(result.damages):
            candidates = scores[index]
            identity, status, score = None, 'unresolved', None
            if len(candidates) == 1 and not blocked[index]:
                identity, score = next(iter(candidates.items()))
                status = 'matched'
            elif not candidates and not blocked[index]:
                if not references and len(features.points) >= 20:
                    identity, status = str(uuid4()), 'baseline'
                elif covered[index]:
                    identity, status = str(uuid4()), 'new'
            associations.append(Association(str(damage.id), identity, status, score))
        # Segmentation splits are ambiguous, not two independent claims on one ID.
        identities = [a.defect_id for a in associations if a.defect_id is not None]
        for association in associations:
            if association.defect_id and identities.count(association.defect_id) > 1:
                association.defect_id, association.status = None, 'unresolved'
        self.store.save(video, result, features, associations)
        return associations
