#!/usr/bin/env python
"""
Extract features from samples obtained from FungusDataset and save them to .npy files. By default in train mode (use train dataset), can be switched to test mode (use test dataset).
"""
import argparse
import logging as log

import numpy as np
import torch
from config import config
from DataLoader import FungusDataset
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


def get_cuda():
    """Return cuda device if available else return cpu."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def extract_features(images, extractor, device):
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


def compute_feature_matrix(loader, extractor, device):
    """Compute feature matrix from entire dataset provided by loader.

    loader - pytorch DataLoader used to draw samples
    extractor - pytorch model used to extract features, if None then AlexNet will be used
    """
    with torch.no_grad():
        feature_matrix = torch.tensor([], dtype=torch.float, device=device)
        labels = torch.tensor([], dtype=torch.long)
        for i, sample in enumerate(tqdm(loader)):
            X = sample['image'].to(device)
            y_true = sample['class']
            X_features = extract_features(X, extractor, device)
            feature_matrix = torch.cat((feature_matrix, X_features), dim=0)
            labels = torch.cat((labels, y_true), dim=0)
    return feature_matrix.cpu().numpy(), labels.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', default=False,
                        action='store_true', help='enable test mode')
    args = parser.parse_args()
    device = get_cuda()
    dataset = FungusDataset(
        dir_with_pngs_and_masks=config['data_path'],
        random_crop_size=125,
        number_of_bg_slices_per_image=2,
        train=not args.test)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=1,
        pin_memory=True)
    feature_matrix, labels = compute_feature_matrix(loader, None, device)
    if args.test:
        filename_prefix = 'test_'
    else:
        filename_prefix = 'train_'
    feature_matrix_filename = filename_prefix + 'feature_matrix.npy'
    labels_filename = filename_prefix + 'labels.npy'
    np.save(feature_matrix_filename, feature_matrix)
    np.save(labels_filename, labels)
