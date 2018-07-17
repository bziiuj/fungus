#!/usr/bin/env python
"""
Perform grid search for the best parameters of the pipeline using train dataset read from .npy files, then save the selected model to `best_model.pkl`.
"""
import logging as log
import tempfile

import numpy as np
import torch
from classification import pipeline
from config import config
from sklearn import model_selection, svm
from sklearn.externals import joblib

if __name__ == '__main__':
    feature_matrix = np.load('train_feature_matrix.npy')
    labels = np.load('train_labels.npy')
    param_grid = [
        {
            'fisher_vector__gmm_samples_number': [1000, 10000],
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['linear']
        },
        {
            'fisher_vector__gmm_samples_number': [1000, 10000],
            'svc__C': [1, 10, 100, 1000],
            'svc__gamma': [0.001, 0.0001],
            'svc__kernel': ['rbf']
        }]
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline.memory = joblib.Memory(cachedir=tmpdir, verbose=0)
        pipeline = model_selection.GridSearchCV(pipeline, param_grid)
        pipeline.fit(feature_matrix, labels)
    log.info(pipeline.best_params_)
    joblib.dump(pipeline, 'best_model.pkl')
