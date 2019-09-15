#!/usr/bin/env python
"""
Extract features from samples obtained from FungusDataset and save them to
.npy files. By default in train mode (use train dataset), can be switched
to test mode (use test dataset).

Also calculates class statistics and saves them to yaml.
"""
import argparse

import numpy as np
import torch
import yaml
from torch.utils import data

from dataset import FungusDataset
from pipeline import features
from util.config import load_config
from util.log import get_logger
from util.log import set_excepthook
from util.path import get_results_path
from util.random import set_seed


def parse_arguments():
    """Builds ArgumentParser and uses it to parse command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'pngs_path', help='absolute path to directory with pngs')
    parser.add_argument(
        'masks_path', help='absolute path to directory with masks')
    parser.add_argument('--test', default=False,
                        action='store_true', help='enable test mode')
    parser.add_argument('--prefix', default='',
                        help='prefix used to aggregate result files in results directory')
    parser.add_argument('--size', default=125, type=int,
                        help='random crop radius')
    parser.add_argument('--config', default='experiments_config.py',
                        help='path to python module with shared experiment configuration')
    parser.add_argument('--prescale', default=None, type=float,
                        help='prescaling factor')
    return parser.parse_args()


def save_class_statistics(path, labels):
    """Calculates class statistics of generated dataset and saves it to yaml file."""
    unique, counts = np.unique(labels, return_counts=True)
    stats = dict(zip(unique.tolist(), counts.tolist()))
    with open(path, mode='w') as f:
        yaml.dump(stats, f)


if __name__ == '__main__':
    logger = get_logger('extract_features')
    set_excepthook(logger)

    args = parse_arguments()
    config = load_config(args.config)
    set_seed(config.seed)
    mode = 'test' if args.test else 'train'
    results_path = get_results_path(
        config.results_path, 'features', args.prefix, mode)
    logger.info('Extracting features for prefix %s in %s mode',
                args.prefix, mode)

    device = features.get_cuda()
    dataset = FungusDataset(
        pngs_dir=args.pngs_path,
        masks_dir=args.masks_path,
        random_crop_size=args.size,
        number_of_bg_slices_per_image=config.number_of_bg_slices_per_image,
        number_of_fg_slices_per_image=config.number_of_fg_slices_per_image,
        train=not args.test,
        prescale=args.prescale)
    loader = data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed(torch.initial_seed()))
    image_patches, feature_matrix, labels = features.compute_feature_matrix(
        loader, device)

    results_path.mkdir(parents=True, exist_ok=True)
    feature_matrix_path = results_path / 'feature_matrix.npy'
    labels_path = results_path / 'labels.npy'
    image_patches_path = results_path / 'image_patches.npy'
    print(feature_matrix.shape)
    np.save(feature_matrix_path, feature_matrix)
    np.save(labels_path, labels)
    np.save(image_patches_path, image_patches)

    statistics_path = results_path / 'statistics.yaml'
    save_class_statistics(statistics_path, labels)
    logger.info('Extraction successfull')
