#!/usr/bin/env python
import argparse
import logging

import numpy as np
import torch
from sklearn import model_selection
from sklearn.externals import joblib

from pipeline import bow
from pipeline import fisher_vector_transformer
from util.config import load_config
from util.log import get_logger
from util.log import set_excepthook
from util.path import get_results_path
from util.random import set_seed


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
    logger = get_logger('train_model')
    set_excepthook(logger)

    args = parse_arguments()
    config = load_config(args.config)
    set_seed(config.seed)
    model = 'bow' if args.bow else 'fv'

    train_features_path = get_results_path(
        config.results_path, 'features', args.prefix, 'train')
    logger.info('Fitting model...')  # add config
    feature_matrix = np.load(train_features_path / 'feature_matrix.npy')
    labels = np.load(train_features_path / 'labels.npy')
    pipeline = bow if args.bow else fisher_vector_transformer
    if args.bow:
        param_grid = config.bow_param_grid
        for i in range(2):
            param_grid[i]['bag_of_words__clusters_number'] = [args.clusters]
    else:
        param_grid = config.fv_param_grid
        for i in range(2):
            param_grid[i]['fisher_vector__gmm_clusters_number'] = [
                args.clusters]
    pipeline = model_selection.GridSearchCV(pipeline, param_grid, n_jobs=24)
    pipeline.fit(feature_matrix, labels)
    logger.info('Best hyperparameters %s', pipeline.best_params_)
    clusters_results_path = get_results_path(
        config.results_path,
        model,
        args.prefix,
        str(args.clusters))
    clusters_results_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, clusters_results_path / 'best_model.pkl')
    test_features_path = get_results_path(
        config.results_path, 'features', args.prefix, 'test')
    feature_matrix = np.load(test_features_path / 'feature_matrix.npy')
    labels = np.load(test_features_path / 'labels.npy')
    logger.info('Accuracy with {} clusters: {}'.format(
        args.clusters,
        pipeline.score(feature_matrix, labels)))
    logger.info('Fitting successfull')
