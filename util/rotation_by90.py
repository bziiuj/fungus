from typing import List

import torchvision.transforms.functional as TF
import random


class RotationBy90(object):

    def __init__(self, angles: List[int] = [0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return TF.rotate(img, angle)
