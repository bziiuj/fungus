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
from torchvision.transforms import Compose

from dataset import FungusDataset
from dataset.normalization import get_normalization_transform
from pipeline import features
from util.augmentation import *
from util.config import load_config
from util.log import get_logger
from util.log import set_excepthook
from util.path import get_results_path
from util.random import set_seed


def parse_arguments():
    """Builds ArgumentParser and uses it to parse command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'imgs_path', help='absolute path to directory with pngs')
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
    parser.add_argument('--augment', action='store_true',
                        help='enable augmentation')
    parser.add_argument('--model', default='alexnet',
                        help='model to use; can be one of alexnet, resnet18')
    parser.add_argument('--reverse', default=False,
                        action='store_true', help='swap train and test subsets')
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
    if args.augment:
        args.prefix += '_aug'
    results_path = get_results_path(
        config.results_path, args.model, args.prefix, mode)
    logger.info(('Extracting features...\n'
                 'prefix: %s\n'
                 'mode: %s\n'
                 'augmentation: %s\n'
                 'model: %s'),
                args.prefix, mode, args.augment, args.model)

    transform = [
        NumpyToTensor(),
        get_normalization_transform(),
    ]
    if args.augment:
        transform.insert(0, NumpyGaussianNoise(sigma=0.01))
    transform = Compose(transform)

    augmentation = None
    if args.augment:
        augmentation = Compose([
            NumpyVerticalFlip(),
            NumpyHorizontalFlip(),
            # NumpyAffineTransform(
            #    scale=(0.8, 1.2),
            #    shear=(np.deg2rad(-15), np.deg2rad(15))
            # )
        ])

    device = features.get_cuda()
    dataset = FungusDataset(
        imgs_dir=args.imgs_path,
        masks_dir=args.masks_path,
        random_crop_size=args.size,
        number_of_fg_slices_per_image=config.number_of_fg_slices_per_image,
        number_of_bg_slices_per_image=config.number_of_bg_slices_per_image,
        train=not args.test,
        transform=transform,
        augmentation=augmentation,
        reverse=args.reverse)
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
