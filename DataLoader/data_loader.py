import numpy as np

from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from normalization import normalize_image


class FungusDataset(Dataset):

    def __init__(
            self,
            paths,
            maps_paths,
            transform=normalize_image,
            scale=None,
            crop=1,
            random_crop_with_background=125,
    ):

        self.fungus_paths = paths
        self.transform = transform
        self.scale = scale
        self.crop = crop
        self.random_crop_with_background = random_crop_with_background
        self.maps_paths = maps_paths

    def __len__(self):
        if self.random_crop_with_background:
            return len(self.fungus_paths) * self.crop * 2
        return len(self.fungus_paths) * self.crop

    def __getitem__(self, idx):
        image, img_class = self.get_image_and_image_class(idx)
        h, w = image.shape

        if self.crop > 1:
            image = self.crop_image(h, idx, image, w)

        scaled = None

        if self.transform:
            if self.scale:
                scaled = (int(h // self.scale), int(w // self.scale))

            if self.random_crop_with_background:
                mask = io.imread(self.maps_paths[int(idx / self.crop / 2)])
                if idx % 2 == 1:
                    where = np.argwhere(mask == 2)
                else:
                    where = np.argwhere(mask == 1)

                center = np.random.uniform(high=where.shape[0])
                y, x = where[int(center)]
                image = image[y - self.random_crop_with_background: y + self.random_crop_with_background,
                              x - self.random_crop_with_background: x + self.random_crop_with_background]

            image = self.transform(image, scaled)

        sample = {'image': image, 'class': img_class, 'bg_or_fg': idx % 2}
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
        if self.random_crop_with_background:
            img_class = self.fungus_paths[int(idx / self.crop / 2)].split('/')[-1][:2]
            image = io.imread(self.fungus_paths[int(idx / self.crop / 2)])
        else:
            img_class = self.fungus_paths[int(idx // self.crop)].split('/')[-1][:2]
            image = io.imread(self.fungus_paths[int(idx // self.crop)])
        return image, img_class


if __name__ == '__main__':
    from glob import glob
    from matplotlib import pyplot as plt

    paths = glob('../pngs/*/*1*')
    maps_paths = glob('../masks/*/*1*')
    # data = FungusDataset(paths, scale=8)
    data = FungusDataset(paths, maps_paths=maps_paths, random_crop_with_background=125)
    # data = FungusDataset(paths)
    # data = FungusDataset(paths, crop=8)
    dl = DataLoader(data, batch_size=2, shuffle=True, num_workers=2)

    for i_batch, sample_batched in enumerate(dl):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['class'])

        # observe 2nd batch and stop.
        if i_batch == 1:
            plt.figure()
            plt.imshow(sample_batched['image'][0].numpy().transpose((1, 2, 0)))
            plt.title(sample_batched['class'][0])
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
