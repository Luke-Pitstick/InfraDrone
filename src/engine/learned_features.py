"""Pretrained SuperPoint + LightGlue matching in original-frame coordinates.

Reference databases must contain SuperPoint descriptors. Models load once per
instance; the first initialization downloads official pretrained weights.
"""
import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint

from src.engine.motion import FrameFeatures


class LearnedFeatures:
    def __init__(self, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.extractor = SuperPoint(max_num_keypoints=1500).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

    @torch.inference_mode()
    def extract(self, image, calibration):
        calibration.check_shape(image.shape)
        if calibration.road_roi is None:
            raise ValueError('Learned alignment requires road_roi')
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device)/255
        features = self.extractor.extract(tensor, resize=None)
        points = features['keypoints'][0].cpu().numpy()
        descriptors = features['descriptors'][0].cpu().numpy()
        left, top, right, bottom = calibration.road_roi
        keep = ((points[:, 0] >= left) & (points[:, 0] < right)
                & (points[:, 1] >= top) & (points[:, 1] < bottom))
        return FrameFeatures(points[keep], descriptors[keep], np.array([image.shape[1], image.shape[0]], np.float32))

    @torch.inference_mode()
    def match(self, first, second):
        if not len(first.points) or not len(second.points):
            return np.empty((0, 2)), np.empty((0, 2))
        inputs = {}
        for name, features in [('image0', first), ('image1', second)]:
            if features.descriptors.dtype != np.float32 or features.descriptors.shape[1] != 256:
                raise ValueError('LightGlue requires a freshly built SuperPoint reference database')
            inputs[name] = {
                key: torch.from_numpy(value).float().to(self.device)[None]
                for key, value in [('keypoints', features.points), ('descriptors', features.descriptors),
                                   ('image_size', features.image_size)]}
        matches = self.matcher(inputs)['matches'][0].cpu().numpy()
        return first.points[matches[:, 0]], second.points[matches[:, 1]]
