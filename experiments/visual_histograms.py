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
from sklearn.externals import joblib


import os  # isort:skip
import sys  # isort:skip

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

from dataset import FungusDataset
from dataset.normalization import denormalize
from pipeline.fisher_vector_transformer import FisherVectorTransformer
from util.config import load_config
from util.log import get_logger
from util.log import set_excepthook
from util.path import get_results_path
from util.random import set_seed


plt.switch_backend('agg')


def generate_bows(feature_matrix, fv, distances):
    bows = []
    for d in range(feature_matrix.shape[0]):
        dist_ = cdist(feature_matrix[d, :], fv.gmm_[0].transpose())
        bows.append(np.histogram(dist_.argmin(axis=1),
                                 bins=np.arange(distances.shape[1] + 1))[0])
    return np.stack(bows)


def plot_similarity_mosaic(distances, patches, filepath):
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
        filename = 'similarity_mosaic_{}.png'.format(str(i))
        plt.savefig(filepath / filename)
        plt.close()


def plot_boxplot(bows, labels, filepath):
    # flierprops = dict(marker='+', markerfacecolor='red')
    plt.figure(figsize=(10, 10), dpi=300)
    for i in range(10):
        i_bows = bows[labels == i, :]
        plt.subplot(2, 5, i + 1)
        sns.boxplot(data=i_bows)
        # plt.boxplot(i_bows, flierprops=flierprops)
        plt.title(FungusDataset.NUMBER_TO_FUNGUS[i])
    plt.savefig(filepath / 'boxplot.png')
    plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='', help='input file prefix')
    parser.add_argument('--bow', default=False,
                        action='store_true', help='enable bow pipeline')
    parser.add_argument('--config', default='experiments_config.py',
                        help='path to python module with shared experiment configuration')
    return parser.parse_args()


if __name__ == '__main__':
    logger = get_logger('visual_histograms')
    set_excepthook(logger)

    args = parse_arguments()
    config = load_config(args.config)
    set_seed(config.seed)
    model = 'bow' if args.bow else 'fv'

    fv = FisherVectorTransformer(gmm_samples_number=5000)
    # svc = svm.SVC(C=10.0, kernel='linear')

    # load train and test data
    train_features_path = get_results_path(
        config.results_path, 'features', args.prefix, 'train')
    train_image_patches = np.load(train_features_path / 'image_patches.npy')
    train_feature_matrix = np.load(train_features_path / 'feature_matrix.npy')
    train_labels = np.load(train_features_path / 'labels.npy')

    test_features_path = get_results_path(
        config.results_path, 'features', args.prefix, 'test')
    test_image_patches = np.load(test_features_path / 'image_patches.npy')
    test_feature_matrix = np.load(test_features_path / 'feature_matrix.npy')
    test_labels = np.load(test_features_path / 'labels.npy')

    # load trained model
    train_results_path = get_results_path(
        config.results_path, model, args.prefix, 'train')
    best_model = joblib.load(train_results_path / 'best_model.pkl')
    transformer_name = 'bag_of_words' if args.bow else 'fisher_vector'
    transformer = best_model.best_estimator_.named_steps[transformer_name]

    # process train and test data with gmm
    train_fv_matrix = transformer.transform(train_feature_matrix)
    test_fv_matrix = transformer.transform(test_feature_matrix)

    # compute distances from train and test to gmm clusters
    train_distances = cdist(
        train_feature_matrix.reshape(-1, 256), transformer.gmm_[0].transpose())
    test_distances = cdist(
        test_feature_matrix.reshape(-1, 256), transformer.gmm_[0].transpose())

    # generate train bow
    train_bows = generate_bows(
        train_feature_matrix, transformer, train_distances)
    test_bows = generate_bows(test_feature_matrix, transformer, test_distances)

    # # compute accuracy
    # svc.fit(train_fv_matrix, train_labels)
    # log.info('Accuracy training {}'.format(
    #     svc.score(train_fv_matrix, train_labels)))
    # log.info('Accuracy test {}'.format(svc.score(test_fv_matrix, test_labels)))

    test_results_path = get_results_path(
        config.results_path, model, args.prefix, 'test')

    # similarity mosaics
    plot_similarity_mosaic(
        train_distances, train_image_patches, train_results_path)
    plot_similarity_mosaic(
        test_distances, test_image_patches, test_results_path)

    # boxplots
    plot_boxplot(train_bows, train_labels, train_results_path)
    plot_boxplot(test_bows, test_labels, test_results_path)
