import argparse
import functools
import os
from pathlib import Path

import numpy as np
from torchvision import transforms

from util.augmentation import NumpyToTensor
from util.io import read_img


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def denormalize(patch):
    means, stds = read_means_and_standard_deviations(
        'tmp/means.npy', 'tmp/stds.npy')
    for i in range(3):
        patch[:, :, i] = np.add(np.multiply(patch[:, :, i], stds[i]), means[i])
    return patch


def get_normalization_transform(path=Path('tmp')):
    means, stds = read_means_and_standard_deviations(
        path / 'means.npy', path / 'stds.npy')
    return transforms.Normalize(means, stds)


def normalize_image(img):
    means, stds = read_means_and_standard_deviations(
        'tmp/means.npy', 'tmp/stds.npy')
    transform = transforms.Compose([
        NumpyToTensor(),
        transforms.Normalize(means, stds),
    ])
    return transform(img)


@functools.lru_cache(maxsize=8)
def read_means_and_standard_deviations(means_path, stds_path):
    return np.load(means_path), np.load(stds_path)


def compute_means_and_standard_deviations(imgs_dir, reverse=False):
    from dataset.img_files import train_paths, test_paths
    paths = test_paths if reverse else train_paths
    means, stds = [], []
    for path in paths:
        full_path = imgs_dir / path
        img = read_img(full_path)
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))
    means = np.array(means)
    stds = np.array(stds)
    return np.mean(means, axis=0), np.std(stds, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'imgs_dir', help='absolute path to directory with imgs')
    parser.add_argument(
        '--reverse', action='store_true', default=False, help='reverse train and test subsets')
    args = parser.parse_args()
    means, stds = compute_means_and_standard_deviations(
        Path(args.imgs_dir), reverse=args.reverse)
    print(means, stds)
    np.save('tmp/means.npy', means)
    np.save('tmp/stds.npy', stds)
