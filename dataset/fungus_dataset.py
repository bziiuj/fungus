import os
import warnings
from enum import IntEnum
from pathlib import Path

import numpy as np
from skimage import transform
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dataset.img_files import test_paths
from dataset.img_files import train_paths
from dataset.normalization import normalize_image
from util.io import read_img
from util.io import read_mask


class ImageSegment(IntEnum):
    NONE = 0
    BACKGROUND = 1
    FOREGROUND = 2


class FungusDataset(Dataset):
    FUNGUS_TO_NUMBER = {
        'CA': 0,
        'CG': 1,
        'CL': 2,
        'CN': 3,
        'CP': 4,
        'CT': 5,
        'MF': 6,
        'SB': 7,
        'SC': 8,
        'BG': 9,
    }
    NUMBER_TO_FUNGUS = {
        0: 'CA',
        1: 'CG',
        2: 'CL',
        3: 'CN',
        4: 'CP',
        5: 'CT',
        6: 'MF',
        7: 'SB',
        8: 'SC',
        9: 'BG',
    }

    def __init__(
            self,
            imgs_dir=None,
            masks_dir=None,
            random_crop_size=125,
            number_of_fg_slices_per_image=8,
            number_of_bg_slices_per_image=0,
            train=True,
            reverse=False,
            transform=None,
            augmentation=None,
    ):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.random_crop_size = random_crop_size
        self.fg_per_img = number_of_fg_slices_per_image
        self.bg_per_img = number_of_bg_slices_per_image
        self.train = train
        self.reverse = reverse
        self.transform = transform
        self.augmentation = augmentation
        if not self.imgs_dir or not self.masks_dir:
            raise AttributeError('Paths to pngs and masks must be provided.')
        if self.train:
            self.paths = train_paths if not self.reverse else test_paths
        else:
            self.paths = test_paths if not self.reverse else train_paths

    def __len__(self):
        return len(self.paths) * (self.fg_per_img + self.bg_per_img)

    def _read_mask(self, image_idx):
        mask_path = self.paths[image_idx]
        mask_path = os.path.join(self.masks_dir, mask_path)
        mask = read_mask(Path(mask_path))
        return mask

    def _read_image_and_class(self, image_idx):
        path = self.paths[image_idx]
        image_class = path.split('/')[-1][:2]
        path = os.path.join(self.imgs_dir, path)
        img = read_img(Path(path))
        return img, image_class

    def _is_foreground_patch(self, sequence_idx):
        return (sequence_idx % (self.bg_per_img + self.fg_per_img)) > self.bg_per_img

    def __getitem__(self, sequence_idx):
        image_idx = sequence_idx // (self.bg_per_img + self.fg_per_img)
        image, image_class = self._read_image_and_class(image_idx)
        mask = self._read_mask(image_idx)

        if self.augmentation:
            image, mask = self.augmentation((image, mask))

        # set appropriate offsets in order to choose only full sized patches
        mask[:self.random_crop_size, :] = ImageSegment.NONE
        mask[-self.random_crop_size:, :] = ImageSegment.NONE
        mask[:, :self.random_crop_size] = ImageSegment.NONE
        mask[:, -self.random_crop_size:] = ImageSegment.NONE

        # prepare a set of coordinates to randomly choose patch center
        if self._is_foreground_patch(sequence_idx):
            if ImageSegment.FOREGROUND in mask:
                where = np.argwhere(mask == ImageSegment.FOREGROUND)
            else:
                warnings.warn(
                    ('Should generate foreground patch from {} class, ' +
                     'but no foreground found in mask. ' +
                     'Toggling to background.').format(image_class))
                where = np.argwhere(mask == ImageSegment.BACKGROUND)
                image_class = 'BG'
        else:
            if ImageSegment.BACKGROUND in mask:
                where = np.argwhere(mask == ImageSegment.BACKGROUND)
                image_class = 'BG'
            else:
                warnings.warn(
                    ('Should generate background patch from {} class, ' +
                     'but no background found in mask. ' +
                     'Toggling to foreground.').format(image_class))
                where = np.argwhere(mask == ImageSegment.FOREGROUND)

        # get a random patch
        center = np.random.uniform(high=where.shape[0])
        y, x = where[int(center)]
        image = image[y - self.random_crop_size:y + self.random_crop_size,
                      x - self.random_crop_size:x + self.random_crop_size,
                      :]

        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'class': self.FUNGUS_TO_NUMBER[image_class],
        }
