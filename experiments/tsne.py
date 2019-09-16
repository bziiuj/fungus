#!/usr/bin/env python
import argparse
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.manifold import TSNE

from pipeline import bow_pipeline
from pipeline import fv_pipeline
from util.config import load_config
from util.log import get_logger
from util.log import set_excepthook
from util.path import get_results_path
from util.random import set_seed

sns.set()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='', help='input file prefix')
    parser.add_argument('--bow', default=False,
                        action='store_true', help='enable bow pipeline')
    parser.add_argument('--config', default='experiments_config.py',
                        help='path to python module with shared experiment configuration')
    parser.add_argument('--clusters', default=50, type=int,
                        help='clusters number to train the model')
    return parser.parse_args()


if __name__ == '__main__':
    logger = get_logger('tsne')
    set_excepthook(logger)

    args = parse_arguments()
    config = load_config(args.config)
    set_seed(config.seed)
    model_type = 'bow' if args.bow else 'fv'

    logger.info('Generating TSNE...')
    train_features_path = get_results_path(
        config.results_path,
        'features',
        args.prefix,
        'train')
    train_feature_matrix = np.load(
        train_features_path / 'feature_matrix.npy')
    test_features_path = get_results_path(
        config.results_path,
        'features',
        args.prefix,
        'test')
    test_feature_matrix = np.load(
        test_features_path / 'feature_matrix.npy')
    clusters_results_path = get_results_path(
        config.results_path,
        model_type,
        args.prefix,
        str(args.clusters))
    model = joblib.load(clusters_results_path /
                        'best_model.pkl').best_estimator_
    step_name = 'bag_of_words' if args.bow else 'fisher_vector'
    transformer = model.named_steps[step_name]
    train_representation = transformer.transform(train_feature_matrix)
    test_representation = transformer.transform(test_feature_matrix)
    train_points = TSNE().fit_transform(train_representation)
    test_points = TSNE().fit_transform(test_representation)
    train_points = pd.DataFrame(train_points, columns=['x', 'y'])
    test_points = pd.DataFrame(test_points, columns=['x', 'y'])
    train_points = train_points.assign(
        train=[True for i in range(train_points.shape[0])])
    test_points = test_points.assign(
        train=[False for i in range(test_points.shape[0])])
    points = train_points.append(test_points)
    sns.relplot(x='x', y='y', hue='train', data=points)
    plt.savefig(clusters_results_path / 'tsne.png')
    plt.close()
    logger.info('TSNE generated for model {} with {} clusters.'.format(
        model_type, str(args.clusters)))
