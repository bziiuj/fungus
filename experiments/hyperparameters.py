#!/usr/bin/env python
import argparse
import logging

import numpy as np
import torch
from sklearn import model_selection
from sklearn.externals import joblib

from pipeline import bow_pipeline
from pipeline import fv_pipeline

import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip


fv_param_grid = [
    {
        'fisher_vector__gmm_samples_number': [10000],
        'fisher_vector__gmm_clusters_number': [5, 10, 20, 50],
        'svc__C': [1, 10, 100, 1000],
        'svc__kernel': ['linear']
    },
    {
        'fisher_vector__gmm_samples_number': [10000],
        'fisher_vector__gmm_clusters_number': [5, 10, 20, 50],
        'svc__C': [1, 10, 100, 1000],
        'svc__gamma': [0.001, 0.0001],
        'svc__kernel': ['rbf']
    }
]

bow_param_grid = [
    {
        'bag_of_words__samples_number': [10000],
        'bag_of_words__clusters_number': [5, 10, 20, 50, 100, 200, 500],
        'svc__C': [1, 10, 100, 1000],
        'svc__kernel': ['linear'],
    },
    {
        'bag_of_words__samples_number': [10000],
        'bag_of_words__clusters_number': [5, 10, 20, 50, 100, 200, 500],
        'svc__C': [1, 10, 100, 1000],
        'svc__gamma': [0.001, 0.0001],
        'svc__kernel': ['rbf']
    }
]

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
    parser.add_argument('--prefix', default='', help='input file prefix')
    parser.add_argument('--bow', default=False,
                        action='store_true', help='enable bow pipeline')
    args = parser.parse_args()

    filename_prefix = '{}{}/{}_{}'.format(args.results_dir,
                                          args.prefix, 'train', args.prefix)
    feature_matrix = np.load('{}_{}'.format(
        filename_prefix, 'feature_matrix.npy'))
    labels = np.load('{}_{}'.format(filename_prefix, 'labels.npy'))

    pipeline = fv_pipeline if not args.bow else bow_pipeline
    param_grid = fv_param_grid if not args.bow else bow_param_grid
    pipeline = model_selection.GridSearchCV(pipeline, param_grid, n_jobs=24)
    pipeline.fit(feature_matrix, labels)
    log.info(pipeline.best_params_)
    model_filename = '{}/{}_{}/best_model.pkl'.format(
        args.results_dir,
        'fv' if not args.bow else 'bow',
        args.prefix,
    )
    joblib.dump(pipeline, model_filename)
