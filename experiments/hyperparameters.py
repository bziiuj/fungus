#!/usr/bin/env python
import argparse
import logging

import numpy as np
import torch
from sklearn import model_selection
from sklearn.externals import joblib

from pipeline.models import pipelines
from util.config import load_config
from util.log import get_logger
from util.log import set_excepthook
from util.path import get_results_path
from util.random import set_seed


def parse_arguments():
    """Builds ArgumentParser and uses it to parse command line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='', help='input file prefix')
    parser.add_argument('--model', default='fv_svc',
                        help='model to use; can be one of fv_svc, bow_svc, fv_rf, bow_rf')
    parser.add_argument('--config', default='experiments_config.py',
                        help='path to python module with shared experiment configuration')
    parser.add_argument('--augment', action='store_true',
                        help='enable augmentation')
    parser.add_argument('--features', default='alexnet',
                        help='which feature extraction method to use; can be one of alexnet, resnet18, inceptionv3')
    return parser.parse_args()


if __name__ == '__main__':
    logger = get_logger('hyperparameters')
    set_excepthook(logger)

    args = parse_arguments()
    config = load_config(args.config)
    set_seed(config.seed)
    features_path = get_results_path(
        config.results_path, args.features, args.prefix, 'train')
    if args.augment:
        args.prefix += '_aug'
        aug_features_path = get_results_path(
            config.results_path, args.features, args.prefix, 'train')
    train_results_path = get_results_path(
        config.results_path, args.features, str(args.model) + '_' + str(args.prefix), 'train')
    train_results_path.mkdir(parents=True, exist_ok=True)
    logger.info('Fitting hyperparameters for prefix %s with %s model',
                args.prefix, args.model)

    feature_matrix = np.load(features_path / 'feature_matrix.npy')
    labels = np.load(features_path / 'labels.npy')
    if args.augment:
        aug_feature_matrix = np.load(aug_features_path / 'feature_matrix.npy')
        aug_labels = np.load(aug_features_path / 'labels.npy')
        feature_matrix = np.concatenate((feature_matrix, aug_feature_matrix))
        labels = np.concatenate((labels, aug_labels))

    pipeline = pipelines[args.model]
    param_grid = config.param_grids[args.model]
    pipeline = model_selection.GridSearchCV(pipeline, param_grid, n_jobs=24)
    pipeline.fit(feature_matrix, labels)
    logger.info('Best hyperparameters %s', pipeline.best_params_)
    joblib.dump(pipeline, train_results_path / 'best_model.pkl')
    logger.info('Hyperparameters fitting successfull')
