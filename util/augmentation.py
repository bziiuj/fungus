import random

import numpy as np
import torch
from skimage.transform import AffineTransform
from skimage.transform import rotate
from skimage.transform import warp
from skimage.util import img_as_ubyte
from skimage.util import random_noise
from torchvision.transforms import Compose
from torchvision.transforms import Lambda
from torchvision.transforms import Normalize


class NumpyGaussianNoise:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, img):
        return random_noise(img, mode='gaussian', var=self.sigma)


class NumpyRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, sample):
        img, mask = sample
        angle = random.choice(self.angles)
        img = rotate(img, angle)
        mask = rotate(mask, angle, order=0, preserve_range=True)
        return img, mask


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
            mask = warp(mask, t, mode='reflect', order=0, preserve_range=True)
        return img, mask


class NumpyToTensor:
    def __call__(self, img):
        img = img.copy()
        img = torch.Tensor(np.ascontiguousarray(
            img.transpose((2, 0, 1)))).float()
        return img
