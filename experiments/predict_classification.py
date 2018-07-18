#!/usr/bin/env python
"""
Read feature matrix and labels from .npy files and classify them. In train
mode use train dataset, fit GMM and then fit SVC, in test mode load best
model obtained from `hyperparameters.py` and perform prediction on test
dataset.
"""
import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import argparse
import logging as log

import numpy as np
from config import config
from pipeline.classification import FisherVectorTransformer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    pipeline = Pipeline(
        steps=[
            ('fisher_vector', FisherVectorTransformer()),
            ('svc', svm.SVC())])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', default=False,
                        action='store_true', help='enable test mode')
    args = parser.parse_args()
    if args.test:
        filename_prefix = 'test_'
    else:
        filename_prefix = 'train_'
    feature_matrix_filename = filename_prefix + 'feature_matrix.npy'
    labels_filename = filename_prefix + 'labels.npy'
    feature_matrix = np.load(feature_matrix_filename)
    labels = np.load(labels_filename)
    if args.test:
        pipeline = joblib.load('best_model.pkl')
    else:
        pipeline.fit(feature_matrix, labels)
    log.info('Accuracy {}'.format(pipeline.score(feature_matrix, labels)))
