import cv2

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.exceptions import EmptyTensorError
from lib.utils import interpolate_dense_features, upscale_positions, preprocess_image


def process_multiscale(image, model, device, soft_threshold=1.01, scales=[.5, 1, 2], preprocessing=None):
    if len(scales) == 3:
        images = [cv2.pyrDown(image), image, cv2.pyrUp(image)]
    else:
        images = [image]

    h_init, w_init, _ = image.shape

    all_keypoints = torch.zeros([3, 0])
    all_descriptors = torch.zeros([
        model.dense_feature_extraction.num_channels, 0
    ])
    all_scores = torch.zeros(0)

    previous_dense_features = None
    for idx, (scale, current_image) in enumerate(zip(scales, images)):
        current_image = preprocess_image(
            current_image,
            preprocessing=preprocessing
        )
        current_image = torch.tensor(
            current_image[np.newaxis, :, :, :].astype(np.float32),
            device=device
        )
        _, _, h_level, w_level = current_image.size()

        dense_features = model.dense_feature_extraction(current_image)
        del current_image

        # Sum the feature maps.
        if previous_dense_features is not None:
            _, _, h, w = dense_features.size()
            dense_features += F.interpolate(
                previous_dense_features, size=[h, w],
                mode='bilinear', align_corners=True
            )
            del previous_dense_features

        # Recover detections.
        detections = model.detection(dense_features)
        fmap_pos = torch.nonzero(detections[0].cpu()).t()
        del detections

        # Soft scores.
        soft_scores = model.soft_detection(dense_features).cpu()
        scores = soft_scores[0][fmap_pos[1, :], fmap_pos[2, :]]
        del soft_scores
        
        # Soft score thresholding.
        mask = (scores > (soft_threshold / (dense_features.shape[-2] * dense_features.shape[-1])))
        fmap_pos = fmap_pos[:, mask]
        scores = scores[mask]
        del mask

        # Recover displacements.
        displacements = model.localization(dense_features)[0].cpu()
        displacements_i = displacements[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        displacements_j = displacements[
            1, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        del displacements

        mask = torch.min(
            torch.abs(displacements_i) < 0.5,
            torch.abs(displacements_j) < 0.5
        )
        fmap_pos = fmap_pos[:, mask]
        scores = scores[mask]
        valid_displacements = torch.stack([
            displacements_i[mask],
            displacements_j[mask]
        ], dim=0)
        del mask, displacements_i, displacements_j

        fmap_keypoints = fmap_pos[1 :, :].float() + valid_displacements
        del fmap_pos, valid_displacements

        try:
            raw_descriptors, _, ids = interpolate_dense_features(
                fmap_keypoints.to(device),
                dense_features[0]
            )
        except EmptyTensorError:
            continue
        fmap_keypoints = fmap_keypoints[:, ids]
        scores = scores[ids]
        previous_dense_features = dense_features
        del dense_features, ids

        keypoints = upscale_positions(fmap_keypoints, scaling_steps=2)
        del fmap_keypoints

        descriptors = F.normalize(raw_descriptors, dim=0)
        del raw_descriptors

        keypoints[0, :] *= h_init / h_level
        keypoints[1, :] *= w_init / w_level

        keypoints = keypoints.cpu()

        keypoints = torch.cat([
            keypoints,
            torch.ones([1, keypoints.size(1)]) * 1 / scale,
        ], dim=0)

        all_keypoints = torch.cat([all_keypoints, keypoints.cpu()], dim=1)
        all_descriptors = torch.cat([all_descriptors, descriptors.cpu()], dim=1)
        all_scores = torch.cat([all_scores, scores.cpu()], dim=0)
        del keypoints, descriptors, scores

    keypoints = all_keypoints.t().numpy()
    scores = all_scores.numpy()
    descriptors = all_descriptors.t().numpy()
    return keypoints, scores, descriptors
