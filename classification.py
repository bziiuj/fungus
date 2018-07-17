#!/usr/bin/env python
"""
Read feature matrix and labels from .npy files and classify them. In train mode use train dataset, fit GMM and then fit SVC on it, in test mode load best model obtained from `hyperparameters.py` and perform prediction on test dataset.
"""
import argparse
import logging as log

import numpy as np
from config import config
from cyvlfeat.fisher import fisher
from cyvlfeat.gmm import gmm
from sklearn import svm
from sklearn.externals import joblib


def fit_gmm(X):
    X = X.reshape(-1, X.shape[2])
    X = X[np.random.choice(
        X.shape[0], config['gmm_train_samples'], replace=False), :]
    means, covars, priors, ll, posteriors = gmm(
        X, n_clusters=config['n_clusters'], init_mode='kmeans')
    means = means.transpose()
    covars = covars.transpose()
    log.debug('{} {} {} {}'.format(
        means.shape, covars.shape, priors.shape, posteriors.shape))
    return (means, covars, priors)


def fisher_vector(features, gmm):
    """Compute Fisher vector from feature vector."""
    means, covars, priors = gmm
    features = features.transpose()
    return fisher(features, means, covars, priors, improved=True)


def compute_fisher_vectors(feature_matrix):
    """Fit GMM and then use it to compute Fisher vector for each image.

    feature_matrix -
    """
    gmm = fit_gmm(feature_matrix)
    fisher_vectors = np.array(
        list(map(lambda x: fisher_vector(x, gmm), feature_matrix)))
    return fisher_vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', default=False,
                        action='store_true', help='enable test mode')
    args = parser.parse_args()
    if args.test:
        filename_prefix = 'test_'
    else:
        filename_prefix = 'train_'
    feature_matrix_filename = filename_prefix + 'feature_matrix.npy'
    labels_filename = filename_prefix + 'labels.npy'
    feature_matrix = np.load(feature_matrix_filename)
    labels = np.load(labels_filename)
    fisher_vectors = compute_fisher_vectors(feature_matrix)
    if args.test:
        clf = joblib.load('best_model.pkl')
    else:
        clf = svm.SVC()
        clf.fit(fisher_vectors, labels)
    log.info('Accuracy {}'.format(clf.score(fisher_vectors, labels)))
