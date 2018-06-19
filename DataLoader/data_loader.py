from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from normalization import normalize_image


class FungusDataset(Dataset):

    def __init__(self, paths, transform=normalize_image, scale=None, crop=1):
        self.fungus_paths = paths
        self.transform = transform
        self.scale = scale
        self.crop = crop

    def __len__(self):
        return len(self.fungus_paths) * self.crop

    def __getitem__(self, idx):
        img_class = self.fungus_paths[int(idx // self.crop)].split('/')[-1][:2]
        image = io.imread(self.fungus_paths[int(idx // self.crop)])
        h, w = image.shape

        if self.crop > 1:
            coeff_start = idx % self.crop
            coeff_stop = coeff_start + 1
            start_pos_h = int(h / self.crop * coeff_start)
            stop_pos_h = int(h / self.crop * coeff_stop)
            start_pos_w = int(w / self.crop * coeff_start)
            stop_pos_w = int(w / self.crop * coeff_stop)
            image = image[start_pos_h:stop_pos_h, start_pos_w:stop_pos_w]

        scaled = None

        if self.transform:
            if self.scale:
                scaled = (int(h // self.scale), int(w // self.scale))

            image = self.transform(image, scaled)

        sample = {'image': image, 'class': img_class}
        return sample


if __name__ == '__main__':
    from glob import glob
    from matplotlib import pyplot as plt

    paths = glob('../pngs/*/*1*')
    data = FungusDataset(paths, scale=8)
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
