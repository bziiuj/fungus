import os
import warnings

import numpy as np
from DataLoader.img_files import test_paths
from DataLoader.img_files import train_paths
from DataLoader.normalization import normalize_image
from skimage import io
from skimage import transform
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


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
            crop=1,
            random_crop_size=125,
            number_of_bg_slices_per_image=0,
            number_of_fg_slices_per_image=8,
            seed=9001,
            train=True,
            pngs_dir=None,
            masks_dir=None
    ):

        self.transform = transform
        self.crop = crop
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
        return len(self.paths) * self.crop * (self.fg_per_img + self.bg_per_img)

    def __getitem__(self, idx):
        image, img_class = self.get_image_and_image_class(idx)
        h, w = image.shape

        if self.crop > 1:
            image = self.crop_image(h, idx, image, w)

        if self.transform:

            mask_path = self.paths[int(
                idx / self.crop / (self.bg_per_img + self.fg_per_img))]
            if self.masks_dir is not None:
                mask_path = os.path.join(self.masks_dir, mask_path)
            mask = io.imread(mask_path)
            if (idx % (self.bg_per_img + self.fg_per_img)) > self.bg_per_img:
                where = np.argwhere(mask == 2)
            elif 1 in mask:
                where = np.argwhere(mask == 1)
                old_class = img_class
                img_class = 'BG'
            else:
                warnings.warn(
                    'No background on image of class {}. Only fg will be returned.'.format(img_class))
                where = np.argwhere(mask == 2)

            cntr = 1000
            y, x = -1, -1
            offset = self.random_crop_size
            while y < offset or y > image.shape[0] - offset or x < offset or x > image.shape[1] - offset:
                center = np.random.uniform(high=where.shape[0])
                y, x = where[int(center)]
                cntr -= 1
                if cntr == 0:
                    warnings.warn(
                        'Not enough background, switching to foreground.')
                    img_class = old_class
                    where = np.argwhere(mask == 2)

            image = image[y - self.random_crop_size: y + self.random_crop_size,
                          x - self.random_crop_size: x + self.random_crop_size]
            image = np.stack((image, image, image), axis=2)
            image = self.transform(image)

        sample = {
            'image': image,
            'class': self.FUNGUS_TO_NUMBER[img_class],
        }

        return sample

    def crop_image(self, h, idx, image, w):
        coeff_start = idx % self.crop
        coeff_stop = coeff_start + 1
        start_pos_h = int(h / self.crop * coeff_start)
        stop_pos_h = int(h / self.crop * coeff_stop)
        start_pos_w = int(w / self.crop * coeff_start)
        stop_pos_w = int(w / self.crop * coeff_stop)
        image = image[start_pos_h:stop_pos_h, start_pos_w:stop_pos_w]
        return image

    def get_image_and_image_class(self, idx):
        img_class = self.paths[int(
            idx / self.crop / (self.bg_per_img + self.fg_per_img))].split('/')[-1][:2]
        path = self.paths[int(
            idx / self.crop / (self.bg_per_img + self.fg_per_img))]
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
