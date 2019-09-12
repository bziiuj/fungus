import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from skimage.transform import AffineTransform
from skimage.transform import rotate
from skimage.transform import warp
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


class NumpyRotation:

    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, sample):
        angle = random.choice(self.angles)
        return rotate(sample[0], angle), rotate(sample[1], angle)


class NumpyVerticalFlip:
    def __call__(self, sample):
        img, mask = sample
        if np.random.uniform() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        return img, mask


class NumpyHorizontalFlip:
    def __call__(self, sample):
        img, mask = sample
        if np.random.uniform() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        return img, mask


class NumpyAffineTransform:
    def __init__(self, scale, shear):
        self.scale = scale
        self.shear = shear

    def __call__(self, sample):
        img, mask = sample
        scale = np.random.uniform(low=self.scale[0], high=self.scale[1])
        shear = np.random.uniform(low=self.shear[0], high=self.shear[1])
        t = AffineTransform(scale=(scale, scale), shear=shear)
        if np.random.uniform() > 0.5:
            img = warp(img, t, mode='reflect')
            mask = warp(mask, t, mode='reflect')
        return img, mask


class NumpyToTensor:
    def __call__(self, img):
        img = img.copy()
        img = torch.Tensor(np.ascontiguousarray(img.transpose((2, 0, 1)))).float()
        return img


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


def get_augmentation_on_numpy_data_img_mask():
    return Compose(
        [
            NumpyRotation(),
            NumpyVerticalFlip(),
            NumpyHorizontalFlip(),
            NumpyAffineTransform(scale=(0.8, 1.2), shear=(np.deg2rad(-15), np.deg2rad(15))),
        ]
    )


def get_augmentation_on_numpy_data_img(noise_sigma=0.1):
    means, stds = read_means_and_standard_deviations('tmp/means.npy', 'tmp/stds.npy')

    return Compose(
        [
            NumpyToTensor(),
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
