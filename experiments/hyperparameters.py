#!/usr/bin/env python
"""
Perform grid search for the best parameters of the pipeline using train
dataset read from .npy files, then save the selected model to
`best_model.pkl`.
"""
import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import argparse
import logging as log

import numpy as np
import torch
from sklearn import model_selection
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from pipeline import FisherVectorTransformer

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='', help='input file prefix')
    args = parser.parse_args()

    pipeline = Pipeline(
        steps=[
            ('fisher_vector', FisherVectorTransformer()),
            ('svc', svm.SVC(probability=True))
        ]
    )

    filename_prefix = 'results/train_'
    if args.prefix:
        filename_prefix += args.prefix + '_'

    feature_matrix = np.load(filename_prefix + 'feature_matrix.npy')
    print(feature_matrix.shape)
    labels = np.load(filename_prefix + 'labels.npy')
    param_grid = [
        {
            'fisher_vector__gmm_samples_number': [1000, 5000, 10000, 100000],
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['linear']
        },
        {
            'fisher_vector__gmm_samples_number': [1000, 5000, 10000, 100000],
            'svc__C': [1, 10, 100, 1000],
            'svc__gamma': [0.001, 0.0001],
            'svc__kernel': ['rbf']
        }]
    pipeline = model_selection.GridSearchCV(pipeline, param_grid, n_jobs=24)
    pipeline.fit(feature_matrix, labels)
    log.info(pipeline.best_params_)
    model_filename = 'results/'
    if args.prefix:
        model_filename += args.prefix + '_'
    model_filename += 'best_model.pkl'
    joblib.dump(pipeline, model_filename)
