import os
import numpy as np

from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from normalization import normalize_image
from img_files import test_fungus_paths
from img_files import test_maps_paths
from img_files import train_fungus_paths
from img_files import train_maps_paths


class FungusDataset(Dataset):
    def __init__(
            self,
            transform=normalize_image,
            scale=None,
            crop=1,
            random_crop_size=125,
            number_of_bg_slices_per_image=0,
            number_of_fg_slices_per_image=8,
            seed=9001,
            train=True,
            dir_with_pngs_and_masks=None
    ):

        self.transform = transform
        self.scale = scale
        self.crop = crop
        self.random_crop_size = random_crop_size
        self.bg_per_img = number_of_bg_slices_per_image
        self.fg_per_img = number_of_fg_slices_per_image
        self.train = train
        self.dir = dir_with_pngs_and_masks
        if self.train:
            self.fungus_paths = train_fungus_paths
            self.maps_paths = train_maps_paths
        else:
            self.fungus_paths = test_fungus_paths
            self.maps_paths = test_maps_paths
        np.random.seed(seed)

        self.fungus_to_number_dict = {
            "CA": 0,
            "CG": 1,
            "CL": 2,
            "CN": 3,
            "CP": 4,
            "CT": 5,
            "MF": 6,
            "SB": 7,
            "SC": 8,
            "BG": 9,
        }

        self.number_to_fungus_dict = {
            0: "CA",
            1: "CG",
            2: "CL",
            3: "CN",
            4: "CP",
            5: "CT",
            6: "MF",
            7: "SB",
            8: "SC",
            9: "BG",
        }

    def __len__(self):
        return len(self.fungus_paths) * self.crop * (self.fg_per_img + self.bg_per_img)

    def __getitem__(self, idx):
        image, img_class = self.get_image_and_image_class(idx)
        h, w = image.shape

        if self.crop > 1:
            image = self.crop_image(h, idx, image, w)

        scaled = None

        if self.transform:
            if self.scale:
                scaled = (int(h // self.scale), int(w // self.scale))

            mask_path = self.maps_paths[int(idx / self.crop / (self.bg_per_img + self.fg_per_img))]
            if self.dir is not None:
                mask_path = os.path.join(self.dir, mask_path)
            mask = io.imread(mask_path)
            if (idx % (self.bg_per_img + self.fg_per_img)) > self.bg_per_img:
                where = np.argwhere(mask == 2)
            elif 1 in mask:
                where = np.argwhere(mask == 1)
                img_class = "BG"
            else:
                Warning("No background on image. Only fg will be returned")
                where = np.argwhere(mask == 2)

            center = np.random.uniform(high=where.shape[0])
            y, x = where[int(center)]
            image = image[y - self.random_crop_size: y + self.random_crop_size,
                          x - self.random_crop_size: x + self.random_crop_size]

            image = self.transform(image, scaled)

        sample = {
            'image': image,
            'class': self.fungus_to_number_dict[img_class],
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
        img_class = self.fungus_paths[int(idx / self.crop / (self.bg_per_img + self.fg_per_img))].split('/')[-1][:2]
        path = self.fungus_paths[int(idx / self.crop / (self.bg_per_img + self.fg_per_img))]
        if self.dir is not None:
            path = os.path.join(self.dir, path)
        image = io.imread(path)
        return image, img_class


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # data = FungusDataset(paths, scale=8)
    # data = FungusDataset(random_crop_size=125, number_of_bg_slices_per_image=2, dir_with_pngs_and_masks=None)
    data = FungusDataset(random_crop_size=125, number_of_bg_slices_per_image=2,
                         dir_with_pngs_and_masks='/home/dawid_rymarczyk/PycharmProjects/fungus', train=True)
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
