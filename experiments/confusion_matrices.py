import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from DataLoader import FungusDataset
from pipeline.classification import FisherVectorTransformer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

matplotlib.use('agg')  # isort:skip


def plot_cnf_matrix(matrix, classes, normalize=False):
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    # legend
    plt.colorbar()
    tick_marks = np.arange(len(classes.keys()))
    plt.xticks(tick_marks, classes.values(), rotation=45)
    plt.yticks(tick_marks, classes.values())
    # numeric values on matrix
    fmt = '.2f' if normalize else 'd'
    tresh = matrix.max() / 2
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if matrix[i, j] > tresh else 'black')

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')


if __name__ == '__main__':
    pipeline = Pipeline(
        steps=[
            ('fisher_vector', FisherVectorTransformer()),
            ('svc', svm.SVC())
        ]
    )
    pipeline = joblib.load('best_model.pkl')

    feature_matrix = np.load('test_feature_matrix.npy')
    y_true = np.load('test_labels.npy')
    y_pred = pipeline.predict(feature_matrix)
    print(pipeline.score(feature_matrix, y_true))
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure()
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS)
    plt.savefig('cnf_matrix.jpg')
    plt.figure()
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS, normalize=True)
    plt.savefig('cnf_matrix_normalized.jpg')
