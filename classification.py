#!/usr/bin/env python
"""
Read feature matrix and labels from .npy files and classify them. In train mode use train dataset, fit GMM and then fit SVC, in test mode load best model obtained from `hyperparameters.py` and perform prediction on test dataset.
"""
import argparse
import logging as log

import numpy as np
from config import config
from cyvlfeat.fisher import fisher
from cyvlfeat.gmm import gmm
from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline


class FisherVectorTransformer(BaseEstimator, TransformerMixin):
    """Fit GMM and compute Fisher vectors"""

    def __init__(self, gmm_clusters_number=10, gmm_samples_number=1000, init_mode='kmeans'):
        self.gmm_clusters_number = gmm_clusters_number
        self.gmm_samples_number = gmm_samples_number
        self.init_mode = init_mode

    def fit(self, X, y=None):
        X = X.reshape(-1, X.shape[2])
        if len(X) < self.gmm_samples_number:
            raise AttributeError(
                'Number of samples must be greater than number of GMM samples')
        X = X[np.random.choice(
            X.shape[0], self.gmm_samples_number, replace=False), :]
        means, covars, priors, ll, posteriors = gmm(
            X, n_clusters=self.gmm_clusters_number, init_mode=self.init_mode)
        means = means.transpose()
        covars = covars.transpose()
        self.gmm_ = (means, covars, priors)
        return self

    def transform(self, X, y=None):
        return np.array(list(map(lambda x: self.__fisher_vector(x), X)))

    def __fisher_vector(self, x):
        """Compute Fisher vector from feature vector x."""
        means, covars, priors = self.gmm_
        x = x.transpose()
        return fisher(x, means, covars, priors, improved=True)


pipeline = Pipeline(
    steps=[
        ('fisher_vector', FisherVectorTransformer()),
        ('svc', svm.SVC())])


if __name__ == '__main__':
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
