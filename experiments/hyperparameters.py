#!/usr/bin/env python
import argparse
import logging

import numpy as np
import torch
from sklearn import model_selection
from sklearn.externals import joblib

from pipeline import bow_pipeline
from pipeline import fv_pipeline
from util.config import load_config
from util.log import get_logger
from util.log import set_excepthook
from util.path import get_results_path
from util.random import set_seed

import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip


def parse_arguments():
    """Builds ArgumentParser and uses it to parse command line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='', help='input file prefix')
    parser.add_argument('--bow', default=False,
                        action='store_true', help='enable bow pipeline')
    parser.add_argument('--config', default='experiments_config.py',
                        help='path to python module with shared experiment configuration')
    return parser.parse_args()


if __name__ == '__main__':
    logger = get_logger('hyperparameters')
    set_excepthook(logger)

    args = parse_arguments()
    config = load_config(args.config)
    set_seed(config.seed)
    model = 'bow' if args.bow else 'fv'
    features_path = get_results_path(
        config.results_path, 'features', args.prefix, 'train')
    train_results_path = get_results_path(
        config.results_path, model, args.prefix, 'train')
    test_results_path = get_results_path(
        config.results_path, model, args.prefix, 'test')
    logger.info('Fitting hyperparameters for prefix %s with %s model',
                args.prefix, model)

    feature_matrix = np.load(features_path / 'feature_matrix.npy')
    labels = np.load(features_path / 'labels.npy')

    pipeline = bow_pipeline if args.bow else fv_pipeline
    param_grid = bow_param_grid if args.bow else fv_param_grid
    pipeline = model_selection.GridSearchCV(pipeline, param_grid, n_jobs=24)
    pipeline.fit(feature_matrix, labels)
    logger.info('Best hyperparameters %s', pipeline.best_params_)
    joblib.dump(pipeline, train_results_path / 'best_model.pkl')
    logger.info('Hyperparameters fitting successfull')
