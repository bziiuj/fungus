#!/usr/bin/env python

import os  # isort:skip
import sys  # isort:skip

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import logging as log

import matplotlib
import numpy as np
import scipy.io as sio
from DataLoader import FungusDataset
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import svm

from pipeline.classification import FisherVectorTransformer  # isort:skip

plt.switch_backend('agg')


def generate_bows(feature_matrix, fv, distances):
    bows = []
    for d in range(feature_matrix.shape[0]):
        dist_ = cdist(feature_matrix[d, :], fv.gmm_[0].transpose())
        bows.append(np.histogram(dist_.argmin(axis=1),
                                 bins=np.arange(distances.shape[1] + 1))[0])
    return np.stack(bows)


# similarity mosaic

def plot_similarity_mosaic(distances, patches, train=False):
    for i in range(distances.shape[1]):
        plt.figure(dpi=300)
        dist = distances[:, i]
        order = np.argpartition(dist, 5 * 5, axis=0)
        print(order.shape)
        closest_patches = patches[order[:25] //
                                  train_feature_matrix.shape[1], :, :, :]
        print(closest_patches.shape)
        for j, patch in enumerate(closest_patches):
            plt.subplot(5, 5, j + 1)
            plt.axis('off')
            print(patch.shape)
            plt.imshow(np.moveaxis(patch, 0, -1))
        if train:
            filename_prefix = 'train_similarity_mosaic_'
        else:
            filename_prefix = 'test_similarity_mosaic_'
        plt.savefig(filename_prefix + str(i) + '.png')
        plt.close()


def plot_boxplot(bows, labels, name):
    flierprops = dict(marker='+', markerfacecolor='red')
    plt.figure(figsize=(10, 10), dpi=300)
    for i in range(10):
        i_bows = bows[labels == i, :]
        plt.subplot(2, 5, i + 1)
        plt.boxplot(i_bows, flierprops=flierprops)
        plt.title(FungusDataset.NUMBER_TO_FUNGUS[i])
    plt.savefig(name)
    plt.close()


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
    train_distances = cdist(
        train_feature_matrix.reshape(-1, 256), fv.gmm_[0].transpose())
    test_distances = cdist(
        test_feature_matrix.reshape(-1, 256), fv.gmm_[0].transpose())

    # get n_samples patches closest to gmm clusters (together with precise location of the closest fragment)
    # n_samples = 7
    # train_cluster_patches = []
    # train_cluster_patches_locations = []
    # for d in range(train_distances.shape[1]):
    #     dist_ = train_distances[:, d]
    #     order_ = np.argsort(dist_)
    #     # take three closest patches
    #     three_ = order_[:n_samples] // train_feature_matrix.shape[1]
    #     train_cluster_patches.append(train_image_patches[three_, :, :, :])
    #     # TODO why modulo?
    #     train_cluster_patches_locations.append(
    #         order_[:n_samples] % train_feature_matrix.shape[1])
    # train_cluster_patches = np.stack(train_cluster_patches)
    # train_cluster_patches_locations = np.stack(train_cluster_patches_locations)

    # generate train bow
    train_bows = generate_bows(train_feature_matrix, fv, train_distances)
    test_bows = generate_bows(test_feature_matrix, fv, test_distances)

    # compute accuracy
    svc.fit(train_fv_matrix, train_labels)
    log.info('Accuracy training {}'.format(
        svc.score(train_fv_matrix, train_labels)))
    log.info('Accuracy test {}'.format(svc.score(test_fv_matrix, test_labels)))

    # similarity mosaics
    # TODO why clipping occurs?
    # TODO should be colorful
    plot_similarity_mosaic(train_distances, train_image_patches, True)
    plot_similarity_mosaic(test_distances, test_image_patches, False)

    # boxplots
    plot_boxplot(train_bows, train_labels, 'train_boxplot.png')
    plot_boxplot(test_bows, test_labels, 'test_boxplot.png')
