import os
import warnings
from enum import IntEnum

import numpy as np
from skimage import io
from skimage import transform
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from DataLoader.img_files import test_paths
from DataLoader.img_files import train_paths
from DataLoader.normalization import normalize_image


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
            transform=normalize_image,
            random_crop_size=125,
            number_of_bg_slices_per_image=0,
            number_of_fg_slices_per_image=8,
            seed=9001,
            train=True,
            pngs_dir=None,
            masks_dir=None
    ):

        self.transform = transform
        self.random_crop_size = random_crop_size
        self.bg_per_img = number_of_bg_slices_per_image
        self.fg_per_img = number_of_fg_slices_per_image
        self.train = train
        self.pngs_dir = pngs_dir
        self.masks_dir = masks_dir
        if self.train:
            self.paths = train_paths
        else:
            self.paths = test_paths
        np.random.seed(seed)

    def __len__(self):
        return len(self.paths) * (self.fg_per_img + self.bg_per_img)

    def _read_mask(self, idx):
        mask_path = self.paths[int(
            idx / (self.bg_per_img + self.fg_per_img))]
        if self.masks_dir is not None:
            mask_path = os.path.join(self.masks_dir, mask_path)
        mask = io.imread(mask_path)
        return mask

    def _is_foreground_patch(self, idx):
        return (idx % (self.bg_per_img + self.fg_per_img)) > self.bg_per_img

    def __getitem__(self, idx):
        image, img_class = self._read_image_and_class(idx)
        mask = self._read_mask(idx)

        # set appropriate offsets in order to choose only full sized patches
        mask[:self.random_crop_size, :] = ImageSegment.NONE
        mask[-self.random_crop_size:, :] = ImageSegment.NONE
        mask[:, :self.random_crop_size] = ImageSegment.NONE
        mask[:, -self.random_crop_size:] = ImageSegment.NONE

        if self._is_foreground_patch(idx):
            if ImageSegment.FOREGROUND in mask:
                where = np.argwhere(mask == ImageSegment.FOREGROUND)
            else:
                warnings.warn(
                    'Should generate foreground patch from {} class, but no foreground found in mask. Toggling to background.'.format(img_class))
                where = np.argwhere(mask == ImageSegment.BACKGROUND)
                img_class = 'BG'
        else:
            if ImageSegment.BACKGROUND in mask:
                where = np.argwhere(mask == ImageSegment.BACKGROUND)
                img_class = 'BG'

            else:
                warnings.warn(
                    'Should generate background patch from {} class, but no background found in mask. Toggling to foreground.'.format(img_class))
                where = np.argwhere(mask == ImageSegment.FOREGROUND)

        center = np.random.uniform(high=where.shape[0])
        y, x = where[int(center)]
        image = image[y - self.random_crop_size:y + self.random_crop_size,
                      x - self.random_crop_size:x + self.random_crop_size,
                      :]
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'class': self.FUNGUS_TO_NUMBER[img_class],
        }

    def _read_image_and_class(self, idx):
        img_class = self.paths[int(
            idx / (self.bg_per_img + self.fg_per_img))].split('/')[-1][:2]
        path = self.paths[int(
            idx / (self.bg_per_img + self.fg_per_img))]
        if self.pngs_dir is not None:
            path = os.path.join(self.pngs_dir, path)
        image = io.imread(path)
        return image, img_class


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # data = FungusDataset(paths, scale=8)
    # data = FungusDataset(random_crop_size=125, number_of_bg_slices_per_image=2, dir_with_pngs_and_masks=None)
    # data = FungusDataset(random_crop_size=125, number_of_bg_slices_per_image=2,
    #                      dir_with_pngs_and_masks='/home/dawid_rymarczyk/PycharmProjects/fungus', train=True)
    # data = FungusDataset(paths)
    # data = FungusDataset(paths, crop=8)
    dl = DataLoader(data, batch_size=32, shuffle=True, num_workers=2)

    for i_batch, sample_batched in enumerate(dl):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['class'])

        # observe 2nd batch and stop.
        if i_batch == 1:
            plt.figure()
            plt.imshow(sample_batched['image'][8].numpy().transpose((1, 2, 0)))
            plt.title(str(sample_batched['class'][8]))
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
