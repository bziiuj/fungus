from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from cyvlfeat.gmm import gmm
from skimage import io
from sklearn.svm import SVC
from torch import Tensor, nn
from torchvision import models


def model_init():
    model = models.alexnet(pretrained=True)
    # new_classifier = nn.Sequential(*list(model.features()))
    # model.classifier = new_classifier
    return model


def image_read(path):
    image = io.imread(path)
    image = np.reshape(
        image, (1, image.shape[2], image.shape[0], image.shape[1]))
    image_tensor = Tensor(np.asarray(image / 256).astype(np.uint8))
    return image_tensor


def extract_features(image_tensor, model):
    data = model.features(image_tensor)
    del model, image_tensor
    data = data.detach().numpy()[0]
    data = data.reshape(data.shape[0], data.shape[2] * data.shape[1])
    data = gmm(data.transpose())
    return data


def prepare_one_image_to_classify(path):
    image_tensor = image_read(path)
    model = model_init()
    data = extract_features(image_tensor, model)
    return data


config_path = Path('config.yml')
config = None
with config_path.open('r') as f:
    config = yaml.load(f)

paths = glob(config['data_path'] + '/*/*')
paths_for_train = []
paths_for_test = []
for path in paths:
    if int(path.split('/')[-1][2:-4]) > 9:
        paths_for_test.append(path)
    else:
        paths_for_train.append(path)

if __name__ == '__main__':
    for path in paths_for_train:
        features_for_svm = prepare_one_image_to_classify(path)
        np.save(path.split('/')[-1][:-4], features_for_svm)
