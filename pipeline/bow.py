import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans


class BOWPooling(BaseEstimator, TransformerMixin):
    def __init__(self, clusters_number=10, samples_number=1000):
        """

        Args:
            clusters_number: Number of GMM views.
            samples_number: Number of points used to fit GMM.
            init_mode: Method to initialize GMM.
        """
        self.clusters_number = clusters_number
        self.samples_number = samples_number
        self.kmeans = None

    def fit(self, X_features, y=None):
        """Fits GMM model to the features.

        Args:
            X_features: Images set of size (n, c, w, h).
        """
        X_features = X_features.reshape(-1, X_features.shape[2])
        if len(X_features) < self.samples_number:
            raise AttributeError(
                'Number of samples must be greater than number of GMM samples')
        indices = np.random.choice(
            X_features.shape[0], self.samples_number, replace=False)
        X_features = X_features[indices, :]

        self.kmeans = KMeans(n_clusters=self.clusters_number, n_jobs=-1)
        self.kmeans.fit(X_features)

        return self

    def transform(self, X, y=None):
        """Transforms features of many images to BoW.

        Args:
            X_features: Features set of size (n, c, w, h).
            y: Labels set of size (n,).

        Returns:
            Fisher vectors set.
        """

        return np.array(list(map(lambda x: self._transform_one(x), X)))

    def _transform_one(self, x):
        """Transforms features of one image to Fisher vector.

        Args:
            x: Features set of size (c, w, h).

        Returns:
            Fisher vector.
        """

        assignment = self.kmeans.predict(x)

        return np.bincount(assignment, minlength=self.clusters_number)
