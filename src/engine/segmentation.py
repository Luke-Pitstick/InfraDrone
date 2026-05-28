"""Road-damage segmentation and crack-branch analysis.

This module wraps a YOLO segmentation model, turns instance masks into
skeletonized crack branches, merges likely split detections, and emits
measurement-rich damage records.
"""

from ultralytics.models import YOLO
import numpy as np
import skimage as ski
import uuid
from pathlib import Path

from skimage.morphology import skeletonize, dilation, disk, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import convolve, distance_transform_edt

from .preprocessing import enhance_for_road_damage
from .models import SegmentationResult, Damage, DamageDimensions, ScalarMeasurement
from .base import BaseEngine
from .constants import (
    UnitTypes,
    DamageType,
    CrackSubtype,
    PotholeSubtype,
    StressRange,
)
from .utils import display_matrix, mask_image_to_segment_arrays

CLASS_MAP = {0: DamageType.CRACK, 1: DamageType.POTHOLE}


class SegmentationEngine(BaseEngine):
    """YOLO-backed engine for segmenting and measuring road-surface damage.

    The engine handles preprocessing, mask extraction, skeleton analysis, branch
    merging, crack subtype classification, and final ``Damage`` construction.
    """

    def __init__(self, model_path: str, img_size: tuple = (640, 640)) -> None:
        """Load a YOLO segmentation model and branch-analysis settings.

        Args:
            model_path: Path to the YOLO segmentation weights file.
            img_size: Inference size passed to YOLO as ``imgsz``.
        """
        super().__init__(unit_type=UnitTypes.px, width_ratio=0, height_ratio=0)
        
        # Engine Settings
        self.stress_range = StressRange.RESIDENTIAL
        self.unit_ratio = 1.0
        

        self.model = YOLO(model_path)
        self.img_size = img_size

        # Combination Settings
        self.combine_threshold = 25
        
        # Crack Subtype Settings
        self.crack_subtype_threshold = 45
        
        # Branch Pruning Settings
        self.branch_pruning_min_length = 20
        self.junction_radius = 5
        
        # Branch Combine Settings
        self.branch_combine_distance_threshold = 50
        self.branch_combine_angle_threshold = 45

    def detect(self, image: np.ndarray) -> list[SegmentationResult]:
        """Run YOLO segmentation and return binary masks with skeletons.

        Args:
            image: BGR image array.

        Returns:
            One ``SegmentationResult`` per instance mask.
        """
        result = self.model([image], imgsz=self.img_size)[0].cpu().numpy()

        segmentation_results = []

        if result.masks is None:
            return segmentation_results

        for i, mask_array in enumerate(result.masks.data):
            # 2D mask array, usually shape: (mask_height, mask_width)
            # Convert the mask to a binary mask
            mask = (mask_array > 0.5).astype(np.uint8)
            skeleton = skeletonize(mask > 0).astype(np.uint8)

            # Detection confidence for this mask
            conf = float(result.boxes.conf[i])

            # Class id for this mask
            cls = int(result.boxes.cls[i])
            class_type = CLASS_MAP[cls]

            segmentation_results.append(SegmentationResult(mask, skeleton, conf, class_type))

        return segmentation_results

    def combine_endpoints(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray,
        endpoint1: np.ndarray,
        endpoint2: np.ndarray,
    ) -> np.ndarray:
        """Merge two masks and draw a line between two skeleton endpoints.

        Args:
            mask1: First binary mask.
            mask2: Second binary mask.
            endpoint1: ``[row, col]`` on the first skeleton.
            endpoint2: ``[row, col]`` on the second skeleton.

        Returns:
            Combined mask with a connecting line drawn between endpoints.
        """
        combined_mask = np.logical_or(mask1, mask2).astype(np.uint8)

        # Update line to use binary dialation to have a realistic radius.

        rr, cc = ski.draw.line(
            int(endpoint1[0]),
            int(endpoint1[1]),
            int(endpoint2[0]),
            int(endpoint2[1]),
        )

        combined_mask[rr, cc] = 1

        return combined_mask

    def get_endpoints(self, mask: np.ndarray) -> np.ndarray:
        """Find skeleton endpoints using 8- or 4-connected neighbor counts.

        Args:
            mask: Binary damage mask.

        Returns:
            Array of shape ``(N, 2)`` with ``[row, col]`` endpoint coordinates.
        """
        # Thin the mask to get pixels
        skeleton_mask = skeletonize(mask)

        # Use scipy to convolve the mask with a 3x3 kernel. To count neighbors of a pixel.
        kernel_8 = np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ]
        )

        kernel_4 = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        )

        # Convolve the mask with the kernel to count neighbors of a pixel.
        convolved_8 = convolve(
            skeleton_mask.astype(np.uint8), kernel_8, mode="constant", cval=0
        )

        # Get the endpoints of the mask.
        endpoints_8 = np.argwhere(skeleton_mask & (convolved_8 == 1))

        if len(endpoints_8) >= 2:
            endpoints = endpoints_8
        else:
            # Use the 4-connected kernel to get the endpoints.
            convolved_4 = convolve(
                skeleton_mask.astype(np.uint8), kernel_4, mode="constant", cval=0
            )
            endpoints = np.argwhere(skeleton_mask & (convolved_4 == 1))

        return endpoints

    def closest_endpoint_pair(
        self, endpoints1: np.ndarray, endpoints2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the closest endpoint pair across two endpoint sets.

        Args:
            endpoints1: Endpoints from the first branch, shape ``(N, 2)``.
            endpoints2: Endpoints from the second branch, shape ``(M, 2)``.

        Returns:
            ``(endpoint1, endpoint2, distance)`` in pixels, or empty arrays and
            ``inf`` if either set is empty.
        """
        if len(endpoints1) == 0 or len(endpoints2) == 0:
            return np.array([]), np.array([]), np.inf

        diffs = endpoints1[:, None, :] - endpoints2[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)

        i, j = np.unravel_index(np.argmin(distances), distances.shape)

        return endpoints1[i], endpoints2[j], distances[i, j]

    def farthest_endpoint_pair(self, endpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the two endpoints with maximum Euclidean separation.

        Args:
            endpoints: Array of shape ``(N, 2)`` with ``[row, col]`` coordinates.

        Returns:
            The two endpoints that span the longest chord of the set.

        Raises:
            ValueError: If fewer than two endpoints are provided.
        """
        if len(endpoints) < 2:
            raise ValueError("Need at least 2 endpoints")
        if len(endpoints) == 2:
            return endpoints[0], endpoints[1]

        diffs = endpoints[:, None, :] - endpoints[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        return endpoints[i], endpoints[j]

    def combine_like_detections(
        self, detections: list[SegmentationResult]
    ) -> list[SegmentationResult]:
        """Merge nearby mask pairs by bridging closest skeleton endpoints.

        Args:
            detections: Segmentation masks from YOLO.

        Returns:
            Combined masks where endpoint distance is below ``combine_threshold``.
        """
        if not detections:
            return []

        used = set()
        combined_results = []

        endpoints_by_index = [
            self.get_endpoints(detection.mask) for detection in detections
        ]

        for i, detection in enumerate(detections):
            if i in used:
                continue

            current_mask = detection.mask.copy()
            current_conf = detection.conf
            current_type = detection.type
            current_endpoints = endpoints_by_index[i]

            for j in range(i + 1, len(detections)):
                if j in used:
                    continue

                next_detection = detections[j]
                next_endpoints = endpoints_by_index[j]

                endpoint1, endpoint2, distance = self.closest_endpoint_pair(
                    current_endpoints,
                    next_endpoints,
                )

                if distance < self.combine_threshold:
                    current_mask = self.combine_endpoints(
                        current_mask,
                        next_detection.mask,
                        endpoint1,
                        endpoint2,
                    )

                    current_conf = max(current_conf, next_detection.conf)
                    used.add(j)

                    # Recompute endpoints because the mask changed.
                    current_endpoints = self.get_endpoints(current_mask)

            used.add(i)
            current_skeleton = skeletonize(current_mask > 0).astype(np.uint8)
            combined_results.append(
                SegmentationResult(
                    current_mask.astype(np.uint8),
                    current_skeleton,
                    current_conf,
                    current_type,
                )
            )

        return combined_results
    
    
    def angle_difference_180(self, a: float, b: float) -> float:
        """Return the smallest angular difference between two undirected line angles.

        Args:
            a: First angle in degrees, in ``[0, 180)``.
            b: Second angle in degrees, in ``[0, 180)``.

        Returns:
            Acute difference in degrees, in ``[0, 90]``.
        """
        return abs((a - b + 90) % 180 - 90)

    def branch_axis_angle(self, mask: np.ndarray) -> float:
        """Estimate overall branch orientation from farthest skeleton endpoints.

        Args:
            mask: Binary branch or skeleton mask.

        Returns:
            Axis angle in degrees in ``[0, 180)``, using ``arctan2(Δrow, Δcol)``.
        """
        endpoints = self.get_endpoints(mask)

        if len(endpoints) < 2:
            pixels = np.argwhere(mask > 0)
            if len(pixels) < 2:
                return 0.0
            ep0, ep1 = self.farthest_endpoint_pair(pixels)
        else:
            ep0, ep1 = self.farthest_endpoint_pair(endpoints)

        row0, col0 = ep0
        row1, col1 = ep1

        angle = np.degrees(np.arctan2(row1 - row0, col1 - col0)) % 180
        return angle


    def local_endpoint_angle(self, mask: np.ndarray, endpoint: np.ndarray, radius: int = 25) -> float:
        """Estimate tangent direction near an endpoint using local PCA on the skeleton.

        Args:
            mask: Binary branch mask.
            endpoint: ``[row, col]`` point on the skeleton.
            radius: Pixel radius for collecting local skeleton points.

        Returns:
            Local direction angle in degrees, in ``[0, 180)``.
        """
        skeleton = skeletonize(mask > 0)
        points = np.argwhere(skeleton)

        if len(points) < 2:
            return 0.0

        distances = np.linalg.norm(points - endpoint, axis=1)
        local_points = points[distances <= radius]

        if len(local_points) < 2:
            local_points = points[np.argsort(distances)[: min(10, len(points))]]

        # Use PCA to estimate local direction.
        centered = local_points - local_points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)

        d_row, d_col = vh[0]
        angle = np.degrees(np.arctan2(d_row, d_col)) % 180

        return angle

    def make_branch_object(self, mask: np.ndarray) -> dict:
        """Build lightweight branch metadata for a binary branch mask.

        Args:
            mask: Binary mask for a single crack branch.

        Returns:
            Dictionary containing the original mask, endpoint coordinates, and
            estimated branch axis angle.
        """
        return {
            "branch": mask,
            "endpoints": self.get_endpoints(mask),
            "angle": self.branch_axis_angle(mask),
        }
        
    def angle_between_points(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Return the undirected angle from ``p1`` to ``p2`` in image coordinates.

        Args:
            p1: Starting point as ``[row, col]``.
            p2: Ending point as ``[row, col]``.

        Returns:
            Angle in degrees in ``[0, 180)``.
        """
        row1, col1 = p1
        row2, col2 = p2
        return np.degrees(np.arctan2(row2 - row1, col2 - col1)) % 180

    def get_closest_endpoint_pair(self, endpoints1: np.ndarray, endpoints2: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the closest endpoint pair across two endpoint arrays.

        This method preserves the older public helper name while delegating to
        ``closest_endpoint_pair``.

        Args:
            endpoints1: Endpoints from the first branch, shape ``(N, 2)``.
            endpoints2: Endpoints from the second branch, shape ``(M, 2)``.

        Returns:
            ``(endpoint1, endpoint2, distance)`` in pixels.
        """
        return self.closest_endpoint_pair(endpoints1, endpoints2)

    def can_combine(
        self,
        branch1: SegmentationResult,
        branch2: SegmentationResult,
    ):
        """Check whether two branches are close enough and aligned to merge.

        Args:
            branch1: First branch with precomputed ``endpoints``.
            branch2: Second branch with precomputed ``endpoints``.

        Returns:
            ``(ok, endpoint1, endpoint2, score)`` where ``score`` is lower when
            branches are a better merge candidate.
        """
        ep1, ep2, distance = self.get_closest_endpoint_pair(
            branch1.endpoints,
            branch2.endpoints,
        )

        if distance >= self.branch_combine_distance_threshold:
            return False, ep1, ep2, np.inf

        angle1 = self.local_endpoint_angle(branch1.skeleton, ep1)
        angle2 = self.local_endpoint_angle(branch2.skeleton, ep2)

        angle_diff = self.angle_difference_180(angle1, angle2)

        if angle_diff >= self.branch_combine_angle_threshold:
            return False, ep1, ep2, np.inf

        score = distance + angle_diff
        return True, ep1, ep2, score

    def combine_branches(
        self,
        branches: list[SegmentationResult],
    ) -> list[SegmentationResult]:
        """Iteratively merge the best eligible branch pair until none remain.

        Args:
            branches: Individual crack branches with endpoints populated.

        Returns:
            Reduced branch list after greedy pairwise merging.
        """
        while True:
            best_pair = None
            best_endpoints = (np.array([]), np.array([]))
            best_score = np.inf

            for idx, branch in enumerate(branches):
                for next_idx, next_branch in enumerate(branches[idx + 1:], start=idx + 1):
                    ok, ep_i, ep_j, score = self.can_combine(
                        branch,
                        next_branch
                    )

                    if ok and score < best_score:
                        best_pair = (idx, next_idx)
                        best_endpoints = (ep_i, ep_j)
                        best_score = score

            if best_pair is None:
                break

            idx, next_idx = best_pair
            ep_i, ep_j = best_endpoints

            merged_mask = self.combine_endpoints(
                branches[idx].mask,
                branches[next_idx].mask,
                ep_i,
                ep_j,
            )

            merged_skeleton = self.combine_endpoints(
                branches[idx].skeleton,
                branches[next_idx].skeleton,
                ep_i,
                ep_j,
            )

            merged_skeleton = skeletonize(merged_skeleton > 0)

            merged_branch = SegmentationResult(
                merged_mask.astype(np.uint8),
                merged_skeleton.astype(np.uint8),
                max(branches[idx].conf, branches[next_idx].conf),
                branches[idx].type,
                branches[idx].num_connections + branches[next_idx].num_connections,
            )

            merged_branch.endpoints = self.get_endpoints(merged_skeleton)

            for remove_idx in sorted([idx, next_idx], reverse=True):
                branches.pop(remove_idx)

            branches.append(merged_branch)

        return branches
    
    def calculate_branch_points(self, skeleton: np.ndarray) -> np.ndarray:
        """Find junction pixels in a skeletonized crack mask.

        Args:
            skeleton: Binary skeleton mask.

        Returns:
            Boolean mask where pixels with three or more 8-connected neighbors
            are marked as branch points.
        """
        kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])

        neighbors = convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)

        # Branch points are skeleton pixels with 3+ neighbors
        branch_points = skeleton & (neighbors >= 3)

        return branch_points
        
    
    def find_branches(self, detections: list[SegmentationResult]) -> tuple[list[SegmentationResult], int]:
        """Split masks into individual branches at skeleton junctions and merge nearby ones.

        Args:
            detections: Combined segmentation masks.

        Returns:
            ``(branches, branch_count)`` after pruning, endpoint extraction, and merging.
        """
        branches = []
        for detection in detections:
            mask = detection.mask
            skeleton = skeletonize(detection.mask)
            branch_points = self.calculate_branch_points(skeleton)
            
            # Junction zone is the area around the branch points to remove them from the skeleton
            junction_zone = dilation(branch_points, disk(5))

            skel_no_branches = skeleton.copy()
            # Remove the junction zone from the skeleton
            skel_no_branches[junction_zone] = False

            # Label each skeleton segment after junctions have been removed.
            labels = label(skel_no_branches, connectivity=2)

            for region in regionprops(labels):
                if region.area <= self.branch_pruning_min_length:
                    continue

                component_skeleton = (labels == region.label)
                component_mask = (
                    dilation(component_skeleton, disk(self.junction_radius))
                    & (mask > 0)
                )
                
                num_connections = len(self.calculate_branch_points(component_skeleton))

                branches.append(
                    SegmentationResult(
                        component_mask.astype(np.uint8),
                        component_skeleton.astype(np.uint8),
                        detection.conf,
                        detection.type,
                        num_connections=num_connections,
                    )
                )
                
        # Calculate the endpoints 
        for branch in branches:
            endpoints = self.get_endpoints(branch.skeleton)
            branch.endpoints = endpoints
            
        # Merge branches that are close to each other.
        branches = self.combine_branches(branches)
        return branches, len(branches)

    def acute_axis_angle(self, angle: float) -> float:
        """Map an undirected axis angle to its acute angle from horizontal.

        Args:
            angle: Axis angle in degrees.

        Returns:
            Acute angle in degrees in ``[0, 90]``.
        """
        angle = angle % 180
        return min(angle, 180 - angle)

    def determine_type(self, segment: np.ndarray, num_connections: int) -> CrackSubtype:
        """Classify a crack branch as longitudinal, transverse, or alligator.

        Uses acute angle to horizontal, assuming the road axis is roughly horizontal
        in the image. High junction counts are treated as alligator cracking.

        Args:
            segment: Branch skeleton or mask.
            num_connections: Junction/branch count used for alligator detection.

        Returns:
            ``CrackSubtype`` for the branch.
        """
        if num_connections > 10:
            return CrackSubtype.ALLIGATOR

        axis_angle = self.branch_axis_angle(segment)
        acute_angle = self.acute_axis_angle(axis_angle)

        if acute_angle >= self.crack_subtype_threshold:
            return CrackSubtype.LONGITUDINAL
        return CrackSubtype.TRANSVERSE

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Enhance an image before YOLO segmentation.

        Args:
            image: Input image array.

        Returns:
            Preprocessed image resized for ``self.img_size``.
        """
        return enhance_for_road_damage(image, size=self.img_size)

    def calculate_thickness(self, mask: np.ndarray) -> ScalarMeasurement:
        """Estimate mean crack thickness from the distance transform.

        Args:
            mask: Binary crack mask.

        Returns:
            Mean crack thickness as a ``DamageMeasurement`` in the engine unit type.
        """
        skel = skeletonize(mask)

        dist = distance_transform_edt(mask) if skel is not None else None

        if dist is None:
            return ScalarMeasurement(value=0, unit=self.unit_type)
        
        # Mean width of the crack in pixels
        mean_width_px = np.mean(2 * dist[skel])
        true_mean_width = mean_width_px * self.unit_ratio

        return ScalarMeasurement(value=true_mean_width, unit=self.unit_type)

    def calculate_length(self, skeleton: np.ndarray) -> ScalarMeasurement:
        """Estimate crack length from the farthest pair of skeleton endpoints.

        Args:
            skeleton: Binary crack skeleton.

        Returns:
            Endpoint-to-endpoint crack length as a ``ScalarMeasurement``.
        """
        endpoints = self.get_endpoints(skeleton)

        if len(endpoints) < 2:
            return ScalarMeasurement(value=0, unit=self.unit_type)

        endpoint1, endpoint2 = self.farthest_endpoint_pair(endpoints)
        distance = float(np.linalg.norm(endpoint2 - endpoint1))
        
        true_distance = distance * self.unit_ratio

        if true_distance is None:
            return ScalarMeasurement(value=0, unit=self.unit_type)

        return ScalarMeasurement(value=true_distance, unit=self.unit_type)
    
    def calculate_area(self, mask: np.ndarray) -> ScalarMeasurement:
        """Calculate crack area from foreground mask pixels.

        Args:
            mask: Binary crack mask.

        Returns:
            Area as a ``Measurement`` after applying the configured unit ratio.
        """
        area = np.sum(mask)
        true_area = area * self.unit_ratio * self.unit_ratio
        return ScalarMeasurement(value=true_area, unit=self.unit_type)

    def calculate_dimensions(self, detection: SegmentationResult) -> DamageDimensions:
        """Compute all geometric dimensions for a segmentation result.

        Args:
            detection: Segmentation result with mask and skeleton arrays.

        Returns:
            ``DamageDimensions`` containing thickness, length, and area measurements.
        """
        mask = detection.mask
        thickness = self.calculate_thickness(mask)
        length = self.calculate_length(detection.skeleton)
        area = self.calculate_area(detection.mask)
        return DamageDimensions(thickness=thickness, length=length, area=area)
    
    def merge_alligator_cracks(self, branches: list[SegmentationResult]) -> Damage:
        """Merge all crack branches into one alligator-crack damage record.

        Args:
            branches: Branches that should be represented as one alligator crack.

        Returns:
            A single ``Damage`` object with unioned mask and skeleton arrays.
        """
        final_mask = np.zeros_like(branches[0].mask)
        final_skeleton = np.zeros_like(branches[0].skeleton)
        final_conf = 0
        final_type = branches[0].type
        
        for branch in branches:
            final_mask = np.logical_or(final_mask, branch.mask)
            final_skeleton = np.logical_or(final_skeleton, branch.skeleton)
            final_conf = max(final_conf, branch.conf)
            
        return Damage(
            id=uuid.uuid4(),
            mask=final_mask,
            skeleton=final_skeleton,
            type=final_type,
            severity=0,
            confidence=final_conf,
            dimensions=self.calculate_dimensions(SegmentationResult(final_mask, final_skeleton, final_conf, final_type)),
            subtype=CrackSubtype.ALLIGATOR,
            stress_range=self.stress_range,
            num_connections=0,
        )

    def process_frame(self, image: np.ndarray) -> list[Damage]:
        """Run the full segmentation pipeline on an image.

        Args:
            image: Raw image array to preprocess and segment.

        Returns:
            Damage records for the detected crack branches.
        """
        damages = []
        preprocessed = self.preprocess(image)
        detections = self.detect(preprocessed)

        # Combine like detections to fix errors in the mask making
        combined_detections = self.combine_like_detections(detections)
        
        # Find branches (individual cracks) for final measurement
        branches, branch_count = self.find_branches(combined_detections)


        for branch in branches:
            dimensions = self.calculate_dimensions(branch)
            subtype = self.determine_type(branch.skeleton, branch_count)
            damages.append(
                Damage(
                    id=uuid.uuid4(),
                    mask=branch.mask,
                    skeleton=branch.skeleton,
                    type=branch.type,
                    severity=0,
                    confidence=branch.conf,
                    dimensions=dimensions,
                    subtype=subtype,
                    stress_range=self.stress_range,
                    num_connections=branch_count,
                )
            )

        return damages    
            
    def _process_frame_test(self, segmentation_results: list[SegmentationResult]) -> list[Damage]:
        """Run post-processing on precomputed segmentation masks.

        This is useful for tests and offline mask inspection because it skips
        preprocessing and YOLO inference.

        Args:
            segmentation_results: Precomputed masks and skeletons.

        Returns:
            Damage records produced from the supplied segmentation results.
        """
        damages = []
        
        # Combine like detections to fix errors in the mask making
        combined_detections = self.combine_like_detections(segmentation_results)
        
        # Find branches (individual cracks) for final measurement
        branches, branch_count = self.find_branches(combined_detections)
        
        
        # Calculate the dimensions and subtype of each branch
        for branch in branches:
            dimensions = self.calculate_dimensions(branch)
            subtype = self.determine_type(branch.skeleton, branch_count)
            
            # Should change eventually, but if any of the cracks are alligator, merge them into one damage and break the loop.
            if subtype == CrackSubtype.ALLIGATOR:
                damages.append(self.merge_alligator_cracks(branches))
                break
            
            
            damages.append(
                Damage(
                    id=uuid.uuid4(),
                    mask=branch.mask,
                    skeleton=branch.skeleton,
                    type=branch.type,
                    severity=0,
                    confidence=branch.conf,
                    dimensions=dimensions,
                    subtype=subtype,
                    stress_range=self.stress_range,
                    num_connections=branch_count,
                )
            )
        
        return damages


if __name__ == "__main__":
    mask_path = Path(
        "/Users/lukepitstick/Projects/Data-Science/InfraDrone/datasets/segmentation/crack_segmentation_dataset/train/masks/CFD_031.jpg"
    )
    masks = mask_image_to_segment_arrays(mask_path)
    
    model_path = Path("/Users/lukepitstick/Projects/Data-Science/InfraDrone/src/ml/models/weights/yolo26s-seg.pt")
    
    segmentation_results = [SegmentationResult(mask, skeletonize(mask > 0), 1, DamageType.CRACK) for mask in masks]
    damages = SegmentationEngine(model_path=str(model_path))._process_frame_test(segmentation_results)

    print(f"Number of damages: {len(damages)}")
    
    for damage in damages:
        print('--------------------------------')
        display_matrix(damage.mask, cmap="gray")
        print(damage)

    
