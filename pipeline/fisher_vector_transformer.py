import logging as log

import numpy as np
from cyvlfeat.fisher import fisher
from cyvlfeat.gmm import gmm
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class FisherVectorTransformer(BaseEstimator, TransformerMixin):
    """Fit GMM and compute Fisher vectors"""

    def __init__(self, gmm_clusters_number=10, gmm_samples_number=10000, init_mode='kmeans'):
        self.gmm_clusters_number = gmm_clusters_number
        self.gmm_samples_number = gmm_samples_number
        self.init_mode = init_mode

    def fit(self, X, y=None):
        X = X.reshape(-1, X.shape[2])
        if len(X) < self.gmm_samples_number:
            raise AttributeError(
                'Number of samples must be greater than the number of GMM samples')
        indices = np.random.choice(
            X.shape[0], self.gmm_samples_number, replace=False)
        X = X[indices, :]
        means, covars, priors, ll, posteriors = gmm(
            X,
            n_clusters=self.gmm_clusters_number,
            init_mode=self.init_mode,
        )
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
