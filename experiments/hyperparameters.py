#!/usr/bin/env python
"""
Perform grid search for the best parameters of the pipeline using train
dataset read from .npy files, then save the selected model to
`best_model.pkl`.
"""
import argparse
import logging

import numpy as np
import torch
from sklearn import model_selection
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from pipeline.fisher_vector_transformer import FisherVectorTransformer

import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip


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
    args = parser.parse_args()

    pipeline = Pipeline(
        steps=[
            ('fisher_vector', FisherVectorTransformer()),
            ('svc', svm.SVC(probability=True))
        ]
    )

    filename_prefix = '{}/{}/{}'.format(args.results_dir,
                                        args.prefix, 'test' if args.test else 'train')
    feature_matrix = np.load('{}_{}'.format(
        filename_prefix, 'feature_matrix.npy'))
    labels = np.load('{}_{}'.format(filename_prefix, 'labels.npy'))
    param_grid = [
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
        }]
    pipeline = model_selection.GridSearchCV(pipeline, param_grid, n_jobs=8)
    pipeline.fit(feature_matrix, labels)
    log.info(pipeline.best_params_)
    model_filename = '{}/{}/best_model.pkl'.format(
        args.results_dir, args.prefix)
    joblib.dump(pipeline, model_filename)
