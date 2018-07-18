#!/usr/bin/env python
"""
Extract features from samples obtained from FungusDataset and save them to
.npy files. By default in train mode (use train dataset), can be switched
to test mode (use test dataset).
"""
import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import argparse

import numpy as np
from config import config
from DataLoader import FungusDataset
from pipeline import features
from torch.utils import data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', default=False,
                        action='store_true', help='enable test mode')
    args = parser.parse_args()
    device = features.get_cuda()
    dataset = FungusDataset(
        dir_with_pngs_and_masks=config['data_path'],
        random_crop_size=125,
        number_of_bg_slices_per_image=2,
        train=not args.test)
    loader = data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=1,
        pin_memory=True)
    feature_matrix, labels = features.compute_feature_matrix(loader, device)
    if args.test:
        filename_prefix = 'test_'
    else:
        filename_prefix = 'train_'
    feature_matrix_filename = filename_prefix + 'feature_matrix.npy'
    labels_filename = filename_prefix + 'labels.npy'
    np.save(feature_matrix_filename, feature_matrix)
    np.save(labels_filename, labels)
