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
import torch
import yaml
from torch.utils import data

from dataset import FungusDataset
from pipeline import features

if __name__ == '__main__':
    SEED = 9001
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'pngs_dir', help='absolute path to directory with pngs')
    parser.add_argument(
        'masks_dir', help='absolute path to directory with masks')
    parser.add_argument(
        'results_dir', help='absolute path to results directory')
    parser.add_argument('--test', default=False,
                        action='store_true', help='enable test mode')
    parser.add_argument('--prefix', default='', help='result filenames prefix')
    parser.add_argument('--size', default=125, type=int,
                        help='random crop radius')
    args = parser.parse_args()
    filename_prefix = '{}/{}/{}'.format(args.results_dir, args.prefix, 'test' if args.test else 'train')

    device = features.get_cuda()
    dataset = FungusDataset(
        pngs_dir=args.pngs_dir,
        masks_dir=args.masks_dir,
        random_crop_size=args.size,
        number_of_bg_slices_per_image=1,
        number_of_fg_slices_per_image=16,
        train=not args.test)
    loader = data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed(torch.initial_seed()))
    image_patches, feature_matrix, labels = features.compute_feature_matrix(
        loader, device)

    np.save('{}_{}'.format(filename_prefix, 'feature_matrix.npy'), feature_matrix)
    np.save('{}_{}'.format(filename_prefix, 'labels.npy'), labels)
    np.save('{}_{}'.format(filename_prefix, 'image_patches.npy'), image_patches)

    unique, counts = np.unique(labels, return_counts=True)
    stats = dict(zip(unique.tolist(), counts.tolist()))
    print(stats)
    with open('{}_{}'.format(filename_prefix, 'stats.yaml'), mode='w') as f:
        yaml.dump(stats, f)
