#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils import data
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image

from dataset import FungusDataset
from pipeline import features

plt.switch_backend('agg')


def save_grid_plot(images):
    plt.figure()
    grid = make_grid(images, padding=100).numpy()
    plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest')
    plt.savefig('check.png')
    plt.close()


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
    parser.add_argument('--test', default=False,
                        action='store_true', help='enable test mode')
    parser.add_argument('--prefix', default='', help='result filenames prefix')
    parser.add_argument('--size', default=125, type=int,
                        help='random crop radius')
    args = parser.parse_args()
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
        num_workers=1,
        pin_memory=True)

    for i, sample in enumerate(loader):
        print(i)
        print(sample['image'].shape)
        save_image(sample['image'][0], 'check.png')

        if i == 0:
            break
