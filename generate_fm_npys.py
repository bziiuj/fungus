import itertools
import logging as log
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from config import config
from cyvlfeat.fisher import fisher
from cyvlfeat.gmm import gmm
from DataLoader import FungusDataset
from skimage import io
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


def get_cuda_if_available():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        return device
    return torch.device('cpu')


def get_feature_extractor():
    model = models.alexnet(pretrained=True)
    extractor = model.features.eval()
    return extractor


def extract_features(images, extractor):
    features = extractor(images)
    N, C, W, H = features.size()
    features = features.reshape(N, C, W * H).transpose_(2, 1)
    log.debug('images {} features before {} after {}'.format(
        images.size(), (N, C, W, H), features.size()))
    return features


def fit_gmm(X):
    log.info('Fitting gmm...')
    X = X.reshape(-1, X.shape[2])
    print(X.shape)
    X = X[np.random.choice(
        X.shape[0], config['gmm_train_samples'], replace=False), :]
    means, covars, priors, ll, posteriors = gmm(
        X, n_clusters=config['n_clusters'], init_mode='kmeans')
    means = means.transpose()
    covars = covars.transpose()
    log.debug('{} {} {} {}'.format(
        means.shape, covars.shape, priors.shape, posteriors.shape))
    return (means, covars, priors)


def compute_fisher_vector(image_features, gmm):
    means, covars, priors = gmm
    image_features = image_features.transpose()
    fv = fisher(image_features, means, covars, priors, improved=True)
    return fv


def train_classifier(X, y):
    log.info('Training classifier...')
    clf = SVC()
    clf.fit(X, y)
    return clf


def get_train_loader():
    dataset = FungusDataset(
        dir_with_pngs_and_masks=config['data_path'],
        random_crop_size=125,
        number_of_bg_slices_per_image=2,
        train=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        num_workers=2, pin_memory=True)
    return loader


def compute_images_features(loader, extractor, device):
    with torch.no_grad():
        images_features = torch.tensor([], dtype=torch.float, device=device)
        labels = torch.tensor([], dtype=torch.long)
        for i, sample in enumerate(tqdm(loader)):
            X = sample['image'].to(device)
            y_true = sample['class']
            X_features = extract_features(X, extractor)
            images_features = torch.cat((images_features, X_features), dim=0)
            labels = torch.cat((labels, y_true), dim=0)
            # if i == 2:
            #    break
    return images_features.cpu().numpy(), labels.numpy()


if __name__ == '__main__':
    log.basicConfig(stream=sys.stdout, level=config['logging_level'])
    device = get_cuda_if_available()
    train_loader = get_train_loader()
    extractor = get_feature_extractor().to(device)
    train_images_features, train_labels = compute_images_features(
        train_loader, extractor, device)
    gmm = fit_gmm(train_images_features)
    fisher_vectors = []
    for image_features in train_images_features:
        fisher_vectors.append(compute_fisher_vector(image_features, gmm))
    fisher_vectors = np.array(fisher_vectors)
    classifier = train_classifier(fisher_vectors, train_labels)
    y_pred = classifier.predict(fisher_vectors)
    acc = np.sum(np.equal(train_labels, y_pred)) / len(train_labels)
    log.info('Training accuracy {}'.format(acc))
