import functools

import numpy as np
from skimage import io
from torchvision import transforms


def normalize_image(img):
    means, stds = read_means_and_standard_deviations(
        'results/means.npy', 'results/stds.npy')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    return transform(img)


@functools.lru_cache(maxsize=8)
def read_means_and_standard_deviations(means_path, stds_path):
    return np.load(means_path), np.load(stds_path)


def compute_means_and_standard_deviations(paths):
    means, stds = [], []
    for path in paths:
        img = io.imread(path)
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))
    means = np.array(means)
    stds = np.array(stds)
    return np.mean(means, axis=0), np.std(stds, axis=0)


if __name__ == '__main__':
    from glob import iglob
    means, stds = compute_means_and_standard_deviations(iglob('../pngs/*/*1*'))
    print(means, stds)
    np.save('results/means.npy', means)
    np.save('results/stds.npy', stds)
