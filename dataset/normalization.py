import argparse
import functools
import os

import numpy as np
from skimage import io
from torchvision import transforms

import os  # isort:skip
import sys  # isort:skip

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip


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


def normalize_image(img):
    means, stds = read_means_and_standard_deviations(
        'tmp/means.npy', 'tmp/stds.npy')
    transform = transforms.Compose([
        transforms.Normalize(means, stds),
    ])
    return transform(img)


@functools.lru_cache(maxsize=8)
def read_means_and_standard_deviations(means_path, stds_path):
    return np.load(means_path), np.load(stds_path)


def compute_means_and_standard_deviations(pngs_dir):
    from dataset.img_files import train_paths
    means, stds = [], []
    for path in train_paths:
        full_path = os.path.join(pngs_dir, path)
        img = io.imread(full_path)
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))
    means = np.array(means)
    stds = np.array(stds)
    return np.mean(means, axis=0) / 255.0, np.std(stds, axis=0) / 255.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'pngs_dir', help='absolute path to directory with pngs')
    args = parser.parse_args()
    means, stds = compute_means_and_standard_deviations(args.pngs_dir)
    print(means, stds)
    np.save('tmp/means.npy', means)
    np.save('tmp/stds.npy', stds)
