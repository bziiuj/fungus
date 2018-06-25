import itertools
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from cyvlfeat.fisher import fisher
from cyvlfeat.gmm import gmm
from skimage import io
from sklearn.svm import SVC
from torch import nn
from torchvision import models


def get_cuda_if_available():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        return device
    return torch.device('cpu')


def read_image(path):
    image = io.imread(path).astype(np.float32)
    # TODO alexnet does not accept grayscale
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)
    else:
        # Move channels to the first dimension
        image = np.moveaxis(image, -1, 0)
    # Normalize to [0, 1]
    image /= 256
    tensor = torch.from_numpy(image)
    return tensor


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
    features = extractor(images)
    return extractor(images)


if __name__ == '__main__':
    config = read_config()
    device = get_cuda_if_available()
    print('Splitting data paths...')
    train_paths, test_paths = split_data_paths(config)
    print('Getting feature extractor...')
    feature_extractor = get_feature_extractor().eval().to(device)
    with torch.no_grad():
        print('Extracting features...')
        train_features = []
        train_labels = []
        for path in train_paths:
            train_labels.append(path.split('/')[-2])
            image = torch.unsqueeze(read_image(path), dim=0).to(device)
            image_features = extract_features(feature_extractor, image)
            _, C, W, H = image_features.size()
            image_features = image_features.reshape(
                -1, C, W * H).transpose_(1, 2)
            train_features.append(image_features)
        train_features = torch.cat(train_features, dim=0)
        print('train_features', train_features.size())
        print('Fitting gmm...')
        means, covars, priors, ll, posteriors = gmm(
            train_features.reshape(-1, train_features.size()[2]), n_clusters=2, init_mode='rand')
        means = np.transpose(means)
        covars = np.transpose(covars)
        print(means.shape, covars.shape, priors.shape, posteriors.shape)
        print('Computing Fisher vectors...')
        fisher_vectors = []
        for features in train_features:
            features = features.cpu().numpy().transpose()
            print(features.shape)
            fv = fisher(features, means, covars, priors)
            fisher_vectors.append(fv)
        fisher_vectors = np.stack(fisher_vectors)
        print(fisher_vectors.shape)
    print('Training classifier...')
    classifier = SVC(C=10)
    classifier.fit(fisher_vectors, train_labels)
    print('Calculating training accuracy...')
    y_pred = classifier.predict(fisher_vectors)
    print(train_labels)
    print(y_pred)
