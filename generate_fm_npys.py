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
    print('Splitting data paths...')
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
    print('Getting feature extractor...')
    model = models.alexnet(pretrained=True)
    extractor = model.features.eval()
    return extractor


def extract_features(images, extractor):
    print('Extracting features...')
    res = []
    for image in images:
        features = extractor(image.unsqueeze(dim=0))
        _, C, W, H = features.size()
        res.append(features.reshape(-1, C, W * H).transpose_(1, 2))
    res = torch.cat(res, dim=0)
    print('train_features', res.size())
    return res


def fit_gmm(X):
    print('Fitting gmm...')
    means, covars, priors, ll, posteriors = gmm(
        X.reshape(-1, X.size()[2]), n_clusters=2, init_mode='rand')
    means = means.transpose()
    covars = covars.transpose()
    print(means.shape, covars.shape, priors.shape, posteriors.shape)
    return (means, covars, priors)


def compute_fisher_vectors(images_features, gmm):
    print('Computing Fisher vectors...')
    means, covars, priors = gmm
    res = []
    for features in images_features:
        features = features.cpu().numpy().transpose()
        print(features.shape)
        fv = fisher(features, means, covars, priors)
        res.append(fv)
    res = np.stack(res)
    print(res.shape)
    return res


def train_classifier(X, y):
    print('Training classifier...')
    clf = SVC()
    clf.fit(X, y)
    return clf


if __name__ == '__main__':
    config = read_config()
    device = get_cuda_if_available()
    train_paths, test_paths = split_data_paths(config)
    extractor = get_feature_extractor().to(device)
    with torch.no_grad():
        train_labels = [path.split('/')[-2] for path in train_paths]
        images = [read_image(path).to(device) for path in train_paths]
        train_features = extract_features(images, extractor)
        gmm = fit_gmm(train_features)
        fisher_vectors = compute_fisher_vectors(train_features, gmm)
    classifier = train_classifier(fisher_vectors, train_labels)
    print('Calculating training accuracy...')
    y_pred = classifier.predict(fisher_vectors)
    print(train_labels)
    print(y_pred)
