import random

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose
from torchvision.transforms import Lambda
from torchvision.transforms import Normalize
from torchvision.transforms import RandomAffine
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomPerspective
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage

from dataset.normalization import read_means_and_standard_deviations


class RotationBy90(object):

    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return TF.rotate(img, angle)


def get_augmentation_on_PIL_image(noise_sigma=0.1):
    means, stds = read_means_and_standard_deviations('tmp/means.npy', 'tmp/stds.npy')

    return Compose(
        [
            RotationBy90(),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            RandomAffine(degrees=(0, 0), scale=(0.8, 1.2), shear=15),
            RandomPerspective(0.25),  # Affine + Perspective = Elastic Transformation
            ToTensor(),
            Normalize(means, stds),
            Lambda(lambda x: x + torch.randn(x.size()) * noise_sigma),
        ]
    )


def get_augmentation_on_tensor_data(noise_sigma=0.1):
    means, stds = read_means_and_standard_deviations('tmp/means.npy', 'tmp/stds.npy')

    return Compose(
        [
            ToPILImage(),
            RotationBy90(),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            RandomAffine(degrees=(0, 0), scale=(0.8, 1.2), shear=15),
            RandomPerspective(0.25),  # Affine + Perspective = Elastic Transformation
            ToTensor(),
            Normalize(means, stds),
            Lambda(lambda x: x + torch.randn(x.size()) * noise_sigma),
        ]
    )


'''
Usage:
    dataset = FungusDataset(
        pngs_dir=args.pngs_path,
        masks_dir=args.masks_path,
        random_crop_size=args.size,
        number_of_bg_slices_per_image=config.number_of_bg_slices_per_image,
        number_of_fg_slices_per_image=config.number_of_fg_slices_per_image,
        train=not args.test,
        transform=get_augmentation_on_tensor_data(),
        )
'''
