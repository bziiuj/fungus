#!/usr/bin/env python
"""
Read feature matrix and labels from .npy files and classify them. In train
mode use train dataset, fit GMM and then fit SVC, in test mode load best
model obtained from `hyperparameters.py` and perform prediction on test
dataset.
"""
import os  # isort:skip
import sys  # isort:skip

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import logging as log

import numpy as np
from sklearn import svm

from pipeline.classification import FisherVectorTransformer  # isort:skip
from scipy.spatial.distance import cdist
import scipy.io as sio

if __name__ == '__main__':
    fv = FisherVectorTransformer(gmm_samples_number=5000)
    svc = svm.SVC(C=10.0, kernel='linear')

    # load train and test data
    train_image_patches = np.load('results/train_image_patches.npy')
    train_feature_matrix = np.load('results/train_feature_matrix.npy')
    train_labels = np.load('results/train_labels.npy')
    test_image_patches = np.load('results/test_image_patches.npy')
    test_feature_matrix = np.load('results/test_feature_matrix.npy')
    test_labels = np.load('results/test_labels.npy')

    # fit gmm with train data
    fv.fit(train_feature_matrix)

    # process train and test data with gmm
    train_fv_matrix = fv.transform(train_feature_matrix)
    test_fv_matrix = fv.transform(test_feature_matrix)

    # compute distances from train and test to gmm clusters
    train_distances = cdist(train_feature_matrix.reshape(-1, 256), fv.gmm_[0].transpose())
    test_distances = cdist(test_feature_matrix.reshape(-1, 256), fv.gmm_[0].transpose())

    # get n_samples patches closest to gmm clusters (together with precise location of the closest fragment)
    n_samples = 7
    train_cluster_patches = []
    train_cluster_patches_locations = []
    for d in range(train_distances.shape[1]):
        dist_ = train_distances[:, d]
        order_ = np.argsort(dist_)
        three_ = order_[:n_samples] // train_feature_matrix.shape[1]  # take three closest patches
        train_cluster_patches.append(train_image_patches[three_, :, :, :])
        train_cluster_patches_locations.append(order_[:n_samples] % train_feature_matrix.shape[1])
    train_cluster_patches = np.stack(train_cluster_patches)
    train_cluster_patches_locations = np.stack(train_cluster_patches_locations)

    # generate train bow
    train_bows = []
    for d in range(train_feature_matrix.shape[0]):
        dist_ = cdist(train_feature_matrix[d, :], fv.gmm_[0].transpose())
        train_bows.append(np.histogram(dist_.argmin(axis=1), bins=np.arange(train_distances.shape[1] + 1))[0])
    train_bows = np.stack(train_bows)
    sio.savemat('results/train_bow.mat', {'cluster_patches': train_cluster_patches,
                                          'labels': train_labels,
                                          'bows': train_bows,
                                          'train_cluster_patches_locations': train_cluster_patches_locations})

    # generate train bow
    test_bows = []
    for d in range(test_feature_matrix.shape[0]):
        dist_ = cdist(test_feature_matrix[d, :], fv.gmm_[0].transpose())
        test_bows.append(np.histogram(dist_.argmin(axis=1), bins=np.arange(test_distances.shape[1] + 1))[0])
    test_bows = np.stack(test_bows)
    sio.savemat('results/test_bow.mat', {'cluster_patches': train_cluster_patches,
                                         'labels': test_labels,
                                         'bows': test_bows,
                                         'train_cluster_patches_locations': train_cluster_patches_locations})

    # fit classifier check accuracy
    svc.fit(train_fv_matrix, train_labels)
    log.info('Accuracy training {}'.format(svc.score(train_fv_matrix, train_labels)))
    log.info('Accuracy test {}'.format(svc.score(test_fv_matrix, test_labels)))
