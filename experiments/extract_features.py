#!/usr/bin/env python
"""
Extract features from samples obtained from FungusDataset and save them to
.npy files. By default in train mode (use train dataset), can be switched
to test mode (use test dataset).

Also calculates class statistics and saves them to yaml.
"""
import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import argparse

import numpy as np
import yaml
from torch.utils import data

from config import config  # isort:skip
from DataLoader import FungusDataset  # isort:skip
from pipeline import features  # isort:skip


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'pngs_dir', help='absolute path to directory with pngs')
    parser.add_argument(
        'masks_dir', help='absolute path to directory with masks')
    parser.add_argument('--test', default=False,
                        action='store_true', help='enable test mode')
    parser.add_argument('--prefix', default='', help='result filenames prefix')
    parser.add_argument('--size', default=125, type=int, help='random crop radius')
    parser.add_argument('--scale', default=None, type=float, help='scale factor')
    args = parser.parse_args()
    device = features.get_cuda()
    dataset = FungusDataset(
        pngs_dir=args.pngs_dir,
        masks_dir=args.masks_dir,
        random_crop_size=args.size,
        number_of_bg_slices_per_image=1,
        number_of_fg_slices_per_image=16,
        train=not args.test,
        scale=args.scale)
    loader = data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=1,
        pin_memory=True)
    feature_matrix, labels = features.compute_feature_matrix(loader, device)
    if args.test:
        filename_prefix = 'results/test_'
    else:
        filename_prefix = 'results/train_'
    if args.prefix:
        filename_prefix += args.prefix + '_'
    feature_matrix_filename = filename_prefix + 'feature_matrix.npy'
    labels_filename = filename_prefix + 'labels.npy'
    np.save(feature_matrix_filename, feature_matrix)
    np.save(labels_filename, labels)

    unique, counts = np.unique(labels, return_counts=True)
    stats = dict(zip(unique.tolist(), counts.tolist()))
    print(stats)
    with open(filename_prefix + 'stats.yaml', mode='w') as f:
        yaml.dump(stats, f)
