import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from cyvlfeat.gmm import gmm
from skimage import io
from sklearn.svm import SVC
from torch import nn
from torchvision import models


def read_image(path):
    image = io.imread(path).astype(np.float32)
    # Move channels to the first dimension
    image = np.moveaxis(image, -1, 0)
    # Normalize to [0, 1]
    image /= 256
    tensor = torch.from_numpy(image)
    return tensor


def prepare_one_image_to_classify(path):
    image_tensor = image_read(path)
    model = model_init()
    data = extract_features(image_tensor, model)
    return data


def read_config():
    config_path = Path('config.yml')
    config = None
    with config_path.open('r') as f:
        config = yaml.load(f)
    return config


def split_data_paths(config):
    paths = glob(config['data_path'] + '/*/*')
    train_paths = []
    test_paths = []
    for path in paths:
        if int(path.split('/')[-1][2:-4]) > 9:
            test_paths.append(path)
        else:
            train_paths.append(path)
    print('Found {} files for training.'.format(
        len(train_paths)), file=sys.stderr)
    print('Found {} files for testing.'.format(
        len(test_paths)), file=sys.stderr)
    return (train_paths, test_paths)


def get_feature_extractor():
    model = models.alexnet(pretrained=True)
    extractor = model.features
    return extractor


def extract_features(extractor, images):
    # We want to return a flat list of features
    return extractor(images).reshape(images.size()[0], -1)


if __name__ == '__main__':
    config = read_config()
    print('Splitting data paths...')
    train_paths, test_paths = split_data_paths(config)
    print('Getting feature extractor...')
    feature_extractor = get_feature_extractor()
    feature_extractor.eval()
    with torch.no_grad():
        print('Extracting features...')
        train_features = []
        for path in train_paths:
            image = torch.unsqueeze(read_image(path), dim=0)
            image_features = extract_features(feature_extractor, image)
            train_features.append(image_features)
        train_features = torch.cat(train_features, dim=0)
        print(train_features.size())
        print('Fitting gmm...')
        priors, means, covars, ll, posteriors = gmm(
            train_features, n_clusters=1, init_mode='rand')
