from ultralytics.models import YOLO
import numpy as np
import skimage as ski
import uuid
from pathlib import Path

from scipy.ndimage import convolve, distance_transform_edt

from .preprocessing import enhance_for_road_damage
from .models import SegmentationResult, Damage, Dimensions, Measurement
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
    def __init__(self, model_path: str, img_size: tuple = (640, 640)) -> None:
        super().__init__(unit_type=UnitTypes.px, width_ratio=0, height_ratio=0)
        
        self.stress_range = StressRange.RESIDENTIAL
        
        self.model = YOLO(model_path)
        self.img_size = img_size

        # Combination Settings
        self.combine_threshold = 10

    def detect(self, image: np.ndarray) -> list[SegmentationResult]:
        # Prediction result for one image
        result = self.model([image], imgsz=self.img_size)[0].cpu().numpy()

        segmentation_results = []

        if result.masks is None:
            return segmentation_results

        for i, mask_array in enumerate(result.masks.data):
            # 2D mask array, usually shape: (mask_height, mask_width)
            # Convert the mask to a binary mask
            mask = (mask_array > 0.5).astype(np.uint8)

            # Detection confidence for this mask
            conf = float(result.boxes.conf[i])

            # Class id for this mask
            cls = int(result.boxes.cls[i])
            class_type = CLASS_MAP[cls]

            segmentation_results.append(SegmentationResult(mask, conf, class_type))

        return segmentation_results

    def _connect_endpoints(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray,
        endpoint1: np.ndarray,
        endpoint2: np.ndarray,
    ) -> np.ndarray:
        combined_mask = np.logical_or(mask1, mask2).astype(np.uint8)

        rr, cc = ski.draw.line(
            int(endpoint1[0]),
            int(endpoint1[1]),
            int(endpoint2[0]),
            int(endpoint2[1]),
        )

        combined_mask[rr, cc] = 1

        return combined_mask

    def _get_endpoints(self, mask: np.ndarray) -> np.ndarray:
        # Thin the mask to get pixels
        skeleton_mask = ski.morphology.skeletonize(mask)

        # Use scipy to convolve the mask with a 3x3 kernel. To count neighbors of a pixel.
        kernel_8 = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])

        kernel_4 = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])

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

    def _closest_endpoint_pair(
        self, endpoints1: np.ndarray, endpoints2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if len(endpoints1) == 0 or len(endpoints2) == 0:
            return np.array([]), np.array([]), np.inf

        diffs = endpoints1[:, None, :] - endpoints2[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)

        i, j = np.unravel_index(np.argmin(distances), distances.shape)

        return endpoints1[i], endpoints2[j], distances[i, j]

    def combine_like_detections(self, detections: list[SegmentationResult]) -> list[SegmentationResult]:
        if not detections:
            return []

        used = set()
        combined_results = []

        endpoints_by_index = [
            self._get_endpoints(detection.mask)
            for detection in detections
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

                endpoint1, endpoint2, distance = self._closest_endpoint_pair(
                    current_endpoints,
                    next_endpoints,
                )

                if distance < self.combine_threshold:
                    current_mask = self._connect_endpoints(
                        current_mask,
                        next_detection.mask,
                        endpoint1,
                        endpoint2,
                    )

                    current_conf = max(current_conf, next_detection.conf)
                    used.add(j)

                    # Recompute endpoints because the mask changed.
                    current_endpoints = self._get_endpoints(current_mask)

            used.add(i)
            combined_results.append(
                SegmentationResult(current_mask.astype(np.uint8), current_conf, current_type)
            )

        return combined_results

    def determine_orientation(self, segment: np.ndarray) -> CrackSubtype:
        # Finds angle of the segment. 
        # If the angle is less than 45 degrees, it is a longitudinal crack
        # If the angle is greater than 45 degrees, it is a transverse crack
        # Returns the crack subtype
        
        endpoints = self._get_endpoints(segment)
        
        angle = np.arctan2(endpoints[1][1] - endpoints[0][1], endpoints[1][0] - endpoints[0][0])
        if angle < np.pi / 4:
            return CrackSubtype.LONGITUDINAL
        elif angle > np.pi / 4:
            return CrackSubtype.TRANSVERSE
        else:
            return CrackSubtype.ALLIGATOR

        
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Return a BGR uint8 image ready for YOLO inference."""
        return enhance_for_road_damage(image, size=self.img_size)

    def post_process(
        self, detections: list[SegmentationResult]
    ) -> list[SegmentationResult]:
        # post-process the detections to get the final result to filter out low confidence detections

        return detections

    def calculate_thickness(self, mask: np.ndarray) -> Measurement:
        skel = ski.morphology.skeletonize(mask)
        
        dist = distance_transform_edt(mask) if skel is not None else None
        
        if dist is None:
            return Measurement(value=0, unit=self.unit_type)
        
        mean_width_px = np.mean(2 * dist[skel])
        
        return Measurement(value=mean_width_px, unit=self.unit_type)
        
    def calculate_length(self, mask: np.ndarray) -> Measurement:
        endpoints = self._get_endpoints(mask)
        
        distance = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        
        if distance is None:
            return Measurement(value=0, unit=self.unit_type)
        
        return Measurement(value=distance, unit=self.unit_type)
        
        
    def calculate_dimensions(self, detection: SegmentationResult) -> Dimensions:
        mask = detection.mask
        thickness = self.calculate_thickness(mask)
        length = self.calculate_length(mask)
        return Dimensions(thickness=thickness, length=length)
        

    def run(self, image: np.ndarray) -> list[Damage]:
        damages = []
        preprocessed = self.preprocess(image)
        detections = self.detect(preprocessed)
        
        # Combine like detections to get a single detection for each crack
        combined_detections = self.combine_like_detections(detections)
        
        for detection in combined_detections:
            dimensions = self.calculate_dimensions(detection)
            subtype = self.determine_orientation(detection.mask)
            damages.append(
                Damage(
                    id=uuid.uuid4(),
                    type=detection.type,
                    severity=0,
                    confidence=detection.conf,
                    dimensions=dimensions,
                    subtype=subtype,
                    stress_range=self.stress_range,
                    num_connections=0,
                )
            )
        
        return damages


if __name__ == "__main__":
    mask_path = Path(
        "/Users/lukepitstick/Projects/Data-Science/InfraDrone/datasets/segmentation/crack_segmentation_dataset/train/masks/CFD_044.jpg"
    )
    masks = mask_image_to_segment_arrays(mask_path)

    thined_masks = []

    for mask in masks:
        thin_mask = ski.morphology.thin(mask)
        thined_masks.append(thin_mask)

        # Use scipy to convolve the mask with a 3x3 kernel. To count neighbors of a pixel.
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # Convolve the mask with the kernel to count neighbors of a pixel.
        convolved = convolve(
            thin_mask.astype(np.uint8), kernel, mode="constant", cval=0
        )

        # Get the endpoints of the mask.
        endpoints = np.argwhere(thin_mask & (convolved == 1))

        print(endpoints)
