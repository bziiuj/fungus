#!/usr/bin/env python

import argparse
import logging

import matplotlib
import numpy as np
import scipy.io as sio
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import svm

from dataset import FungusDataset
from dataset.normalization import denormalize
from pipeline.fisher_vector_transformer import FisherVectorTransformer

import os  # isort:skip
import sys  # isort:skip

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip


plt.switch_backend('agg')


def generate_bows(feature_matrix, fv, distances):
    bows = []
    for d in range(feature_matrix.shape[0]):
        dist_ = cdist(feature_matrix[d, :], fv.gmm_[0].transpose())
        bows.append(np.histogram(dist_.argmin(axis=1),
                                 bins=np.arange(distances.shape[1] + 1))[0])
    return np.stack(bows)


def plot_similarity_mosaic(distances, patches, filename_prefix):
    for i in range(distances.shape[1]):
        plt.figure(dpi=300)
        dist = distances[:, i]
        order = np.argpartition(dist, 5 * 5, axis=0)
        closest_patches = patches[order[:25] //
                                  train_feature_matrix.shape[1], :, :, :]
        for j, patch in enumerate(closest_patches):
            plt.subplot(5, 5, j + 1)
            plt.axis('off')
            patch = np.moveaxis(patch, 0, -1)
            # print('pre')
            # print(patch)
            patch = denormalize(patch)
            # print('post')
            # print(patch)
            plt.imshow(patch)
        plt.savefig('{}_similarity_mosaic_{}.png'.format(
            filename_prefix, str(i)))
        plt.close()


def plot_boxplot(bows, labels, name):
    # flierprops = dict(marker='+', markerfacecolor='red')
    plt.figure(figsize=(10, 10), dpi=300)
    for i in range(10):
        i_bows = bows[labels == i, :]
        plt.subplot(2, 5, i + 1)
        sns.boxplot(data=i_bows)
        # plt.boxplot(i_bows, flierprops=flierprops)
        plt.title(FungusDataset.NUMBER_TO_FUNGUS[i])
    plt.savefig(name)
    plt.close()


if __name__ == '__main__':
    log = logging.getLogger('fungus')
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler('tmp/fungus.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    log.addHandler(ch)
    log.addHandler(fh)

    SEED = 9001
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results_dir', help='absolute path to results directory')
    parser.add_argument('--prefix')
    args = parser.parse_args()

    fv = FisherVectorTransformer(gmm_samples_number=5000)
    svc = svm.SVC(C=10.0, kernel='linear')

    # load train and test data
    train_filename_prefix = '{}/{}/{}'.format(
        args.results_dir, args.prefix, 'train')
    train_image_patches = np.load('{}_{}'.format(
        train_filename_prefix, 'image_patches.npy'))
    train_feature_matrix = np.load('{}_{}'.format(
        train_filename_prefix, 'feature_matrix.npy'))
    train_labels = np.load('{}_{}'.format(train_filename_prefix, 'labels.npy'))

    test_filename_prefix = '{}/{}/{}'.format(
        args.results_dir, args.prefix, 'test')
    test_image_patches = np.load('{}_{}'.format(
        test_filename_prefix, 'image_patches.npy'))
    test_feature_matrix = np.load('{}_{}'.format(
        test_filename_prefix, 'feature_matrix.npy'))
    test_labels = np.load('{}_{}'.format(test_filename_prefix, 'labels.npy'))

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

    # generate train bow
    train_bows = generate_bows(train_feature_matrix, fv, train_distances)
    test_bows = generate_bows(test_feature_matrix, fv, test_distances)

    # compute accuracy
    svc.fit(train_fv_matrix, train_labels)
    log.info('Accuracy training {}'.format(
        svc.score(train_fv_matrix, train_labels)))
    log.info('Accuracy test {}'.format(svc.score(test_fv_matrix, test_labels)))

    # similarity mosaics
    plot_similarity_mosaic(
        train_distances, train_image_patches, train_filename_prefix)
    plot_similarity_mosaic(
        test_distances, test_image_patches, test_filename_prefix)

    # boxplots
    plot_boxplot(train_bows, train_labels, '{}_{}'.format(
        train_filename_prefix, 'boxplot.png'))
    plot_boxplot(test_bows, test_labels, '{}_{}'.format(
        test_filename_prefix, 'boxplot.png'))
