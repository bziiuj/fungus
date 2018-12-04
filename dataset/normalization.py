import argparse
import functools
import os

import numpy as np
from skimage import io
from torchvision import transforms

train_paths = [
    'CA/CA1.png',
    'CA/CA10.png',
    'CA/CA2.png',
    'CA/CA3.png',
    'CA/CA4.png',
    'CA/CA5.png',
    'CA/CA6.png',
    'CA/CA7.png',
    'CA/CA8.png',
    'CA/CA9.png',
    'CG/CG1.png',
    'CG/CG10.png',
    'CG/CG2.png',
    'CG/CG3.png',
    'CG/CG4.png',
    'CG/CG5.png',
    'CG/CG6.png',
    'CG/CG7.png',
    'CG/CG8.png',
    'CG/CG9.png',
    'CL/CL1.png',
    'CL/CL10.png',
    'CL/CL2.png',
    'CL/CL3.png',
    'CL/CL4.png',
    'CL/CL5.png',
    'CL/CL6.png',
    'CL/CL7.png',
    'CL/CL8.png',
    'CL/CL9.png',
    'CN/CN1.png',
    'CN/CN10.png',
    'CN/CN2.png',
    'CN/CN3.png',
    'CN/CN4.png',
    'CN/CN6.png',
    'CN/CN7.png',
    'CN/CN8.png',
    'CN/CN9.png',
    'CP/CP1.png',
    'CP/CP10.png',
    'CP/CP2.png',
    'CP/CP3.png',
    'CP/CP4.png',
    'CP/CP5.png',
    'CP/CP6.png',
    'CP/CP7.png',
    'CP/CP8.png',
    'CP/CP9.png',
    'CT/CT1.png',
    'CT/CT10.png',
    'CT/CT2.png',
    'CT/CT3.png',
    'CT/CT4.png',
    'CT/CT5.png',
    'CT/CT6.png',
    'CT/CT7.png',
    'CT/CT8.png',
    'CT/CT9.png',
    'MF/MF1.png',
    'MF/MF10.png',
    'MF/MF2.png',
    'MF/MF3.png',
    'MF/MF4.png',
    'MF/MF5.png',
    'MF/MF6.png',
    'MF/MF7.png',
    'MF/MF8.png',
    'MF/MF9.png',
    'SB/SB1.png',
    'SB/SB10.png',
    'SB/SB2.png',
    'SB/SB3.png',
    'SB/SB4.png',
    'SB/SB5.png',
    'SB/SB6.png',
    'SB/SB7.png',
    'SB/SB8.png',
    'SB/SB9.png',
    'SC/SC1.png',
    'SC/SC10.png',
    'SC/SC2.png',
    'SC/SC3.png',
    'SC/SC4.png',
    'SC/SC5.png',
    'SC/SC6.png',
    'SC/SC7.png',
    'SC/SC8.png',
    'SC/SC9.png',
]


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


def compute_means_and_standard_deviations(pngs_dir):
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
    np.save('results/means.npy', means)
    np.save('results/stds.npy', stds)
