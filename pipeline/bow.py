import numpy as np
from cyvlfeat.kmeans import kmeans
from cyvlfeat.kmeans import kmeans_quantize
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class BagOfWordsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clusters_number=10, samples_number=1000, algorithm='ANN'):
        """
        Args:
            clusters_number: number of clusters to find
            samples_number: number of samples used to find clusters
        """
        self.clusters_number = clusters_number
        self.samples_number = samples_number
        self.centers = None
        self.algorithm = algorithm

    def fit(self, X, y=None):
        """Fit clusters centers using KMeans.

        Args:
            X - 3D array with shape (number of samples, image width * height, number of channels)
            y - unused
        """
        X = X.reshape(-1, X.shape[2])
        if len(X) < self.samples_number:
            raise AttributeError(
                'Number of samples must be greater than declared in initialization')
        indices = np.random.choice(
            X.shape[0], self.samples_number, replace=False)
        X = X[indices, :]
        self.centers = kmeans(X, self.clusters_number,
                              algorithm=self.algorithm)
        return self

    def transform(self, X, y=None):
        """Find closest clusters for each sample.

        Args:
            X: Features set of size (n, w * h, c)

        Returns:
            Array with cluster labels of shape (n,).
        """
        return np.array(list(map(lambda x: self.__transform_one(x), X)))

    def __transform_one(self, x):
        """Compute bag of words bincount per one image.

        Args:
            x: Features set of size (w * h, c).

        Returns:
            Bincount.
        """
        assignment = kmeans_quantize(x, self.centers, algorithm=self.algorithm)
        return np.bincount(assignment, minlength=self.clusters_number)
