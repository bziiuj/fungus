"""
Functions used to extract features from fungus images.
"""
import logging as log

import numpy as np
import torch
from torchvision import models


def get_cuda():
    """Return cuda device if available else return cpu."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def extract_features(images, device, extractor=None):
    """Extract features from a set of images

    images -- set of images with shape N, C, W, H
    extractor -- pytorch model to be used as an extractor. If None then
    AlexNet will be used.
    """
    if not extractor:
        extractor = models.alexnet(pretrained=True).features.eval().to(device)
    features = extractor(images)
    N, C, W, H = features.size()
    features = features.reshape(N, C, W * H).transpose_(2, 1)
    log.debug('images {} features before {} after {}'.format(
        images.size(), (N, C, W, H), features.size()))
    return features


def compute_feature_matrix(loader, device, extractor=None):
    """Compute feature matrix from entire dataset provided by loader.

    loader - pytorch DataLoader used to draw samples
    extractor - pytorch model used to extract features, if None then AlexNet will be used
    """
    with torch.no_grad():
        image_patches = torch.tensor([], dtype=torch.uint8, device=device)
        feature_matrix = torch.tensor([], dtype=torch.float, device=device)
        labels = torch.tensor([], dtype=torch.long)
        for i, sample in enumerate(loader):
            print(i)
            orig_X = sample['orig_image'].to(device)
            X = sample['image'].to(device)
            y_true = sample['class']
            X_features = extract_features(X, device, extractor)

            feature_matrix = torch.cat((feature_matrix, X_features), dim=0)
            image_patches = torch.cat((image_patches, orig_X), dim=0)
            labels = torch.cat((labels, y_true), dim=0)

    return image_patches.cpu().numpy(), feature_matrix.cpu().numpy(), labels.numpy()
