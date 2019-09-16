#!/usr/bin/env python
import argparse
import logging

import matplotlib
import numpy as np
import scipy.io as sio
import seaborn as sns
import torch
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.externals import joblib

from dataset import FungusDataset
from dataset.normalization import denormalize
from pipeline.fisher_vector_transformer import FisherVectorTransformer
from util.config import load_config
from util.log import get_logger
from util.log import set_excepthook
from util.path import get_results_path
from util.random import set_seed

plt.switch_backend('agg')
sns.set()


def plot_similarity_mosaic(distances, patches, filepath):
    p = []
    indices = [
        [0, 3, 6, 10, 11, 14, 16, 20, 22, 23],
        [0, 2, 3, 6, 7, 9, 10, 17, 20, 21],
        [0, 5, 6, 7, 8, 10, 15, 17, 19, 23],
        [0, 1, 2, 5, 7, 10, 15, 22, 23, 24],
        [0, 4, 5, 9, 11, 15, 20, 22, 23, 24],
        [0, 2, 5, 6, 8, 10, 14, 16, 17, 19],
        [0, 6, 8, 9, 13, 15, 20, 21, 22, 23],
        [0, 1, 6, 7, 10, 17, 18, 19, 20, 21],
        [0, 5, 6, 8, 14, 15, 17, 18, 19, 20],
        [0, 2, 3, 5, 6, 10, 11, 16, 17, 24]
    ]
    for i in range(distances.shape[1]):
        p.append([])
        plt.figure(dpi=300)
        dist = distances[:, i]
        order = np.argpartition(dist, 5 * 5, axis=0)
        closest_patches = patches[order[:25] //
                                  train_feature_matrix.shape[1], :, :, :]
        p[-1].extend(closest_patches[indices[i]])
        for j, patch in enumerate(closest_patches):
            plt.subplot(5, 5, j + 1)
            plt.axis('off')
            patch = np.moveaxis(patch, 0, -1)
            patch = denormalize(patch)
            plt.imshow(patch)
        filename = 'similarity_mosaic_{}.png'.format(str(i))
        plt.savefig(filepath / filename)
        plt.close()
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.1, hspace=0.1)
    for i in range(10):
        for j in range(len(p[i])):
            plt.subplot(gs[i * 10 + j])
            plt.axis('off')
            t = np.moveaxis(p[i][j], 0, -1)
            t = denormalize(t)
            plt.imshow(t)
    plt.savefig(filepath / 'similarity_mosaic.png')
    plt.close()


def plot_boxplot(bows, labels, filepath):
    f, axes = plt.subplots(5, 2, figsize=(25, 40), sharex=True)
    for i in range(len(FungusDataset.NUMBER_TO_FUNGUS.values())):
        i_bows = bows[labels == i, :]
        ax = axes[i // 2, i % 2]
        sns.boxplot(data=i_bows, orient='h', ax=ax)
        plt.title(FungusDataset.NUMBER_TO_FUNGUS[i])
    plt.tight_layout()
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


def load_all(config, args, mode):
    features_path = get_results_path(
        config.results_path, 'features', args.prefix, mode)
    image_patches = np.load(features_path / 'image_patches.npy')
    feature_matrix = np.load(features_path / 'feature_matrix.npy')
    labels = np.load(features_path / 'labels.npy')
    return image_patches, feature_matrix, labels


if __name__ == '__main__':
    logger = get_logger('visual_histograms')
    set_excepthook(logger)

    args = parse_arguments()
    config = load_config(args.config)
    set_seed(config.seed)
    model_type = 'bow' if args.bow else 'fv'
    CLUSTERS_NUM = 10

    # train_image_patches, train_feature_matrix, train_labels = load_all(config, args, 'train')
    test_image_patches, test_feature_matrix, test_labels = load_all(
        config, args, 'test')
    model_path = get_results_path(
        config.results_path, model_type, args.prefix, str(CLUSTERS_NUM))
    model = joblib.load(model_path / 'best_model.pkl').best_estimator_
    transformer_name = 'bag_of_words' if args.bow else 'fisher_vector'
    transformer = model.named_steps[transformer_name]

    # train_points = transformer.transform(train_feature_matrix)
    test_points = transformer.transform(test_feature_matrix)

    # compute distances from train and test to gmm clusters
    # train_distances = cdist(
    # train_feature_matrix.reshape(-1, 256), transformer.gmm_[0].transpose())
    # test_distances = cdist(
    #     test_feature_matrix.reshape(-1, 256), transformer.gmm_[0].transpose())

    # plot_similarity_mosaic(test_distances, test_image_patches, model_path)

    test_bows = transformer.transform(test_feature_matrix)
    plot_boxplot(test_bows, test_labels, model_path)
