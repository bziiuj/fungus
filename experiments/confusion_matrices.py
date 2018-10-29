import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import itertools

from config import config  # isort:skip
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from DataLoader import FungusDataset
from pipeline.classification import FisherVectorTransformer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

plt.switch_backend('agg')


def plot_cnf_matrix(matrix, classes, title, normalize=False):
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    # legend
    plt.colorbar()
    tick_marks = np.arange(len(classes.keys()))
    plt.xticks(tick_marks, classes.values(), rotation=45)
    plt.yticks(tick_marks, classes.values())
    # numeric values on matrix
    # fmt = '.2f' if normalize else 'd'
    fmt = '.2f'
    tresh = matrix.max() / 2
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if matrix[i, j] > tresh else 'black')

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')


def probability_confusion_matrix(y_true, y_pred, probabilities, classes):
    n_classes = len(classes.keys())
    dim = (n_classes, n_classes)
    matrix = np.zeros(dim)
    count_matrix = np.zeros(dim)
    for i in range(len(y_true)):
        matrix[y_true[i], y_pred[i]] += probabilities[i, y_pred[i]]
        count_matrix[y_true[i], y_pred[i]] += 1
    tmp = np.divide(matrix, count_matrix)
    return tmp


def plot_accuracy_bars(cnf_matrix, classes, title):
    print(np.diag(cnf_matrix))
    print(np.sum(cnf_matrix, axis=1))
    accuracy = np.diag(cnf_matrix) / np.sum(cnf_matrix, axis=1)
    plt.title(title)
    plt.bar(classes.values(), accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Classes')


if __name__ == '__main__':
    pipeline = Pipeline(
        steps=[
            ('fisher_vector', FisherVectorTransformer()),
            ('svc', svm.SVC())
        ]
    )
    pipeline = joblib.load('{}/2_3_250_50p_best_model.pkl'.format(config['analysis_path']))

    # train
    feature_matrix = np.load('{}/train_feature_matrix.npy'.format(config['analysis_path']))
    y_true = np.load('{}/train_labels.npy'.format(config['analysis_path']))
    y_pred = pipeline.predict(feature_matrix)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure()
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS,
                    'Train cnf matrix')
    plt.savefig('{}/train_cnf_matrix.jpg'.format(config['analysis_path']))
    plt.figure()
    plot_accuracy_bars(
        cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS, 'Train accuracy')
    plt.savefig('{}/train_accuracy_bars.jpg'.format(config['analysis_path']))
    plt.figure()
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS,
                    'Train normalized cnf matrix', normalize=True)
    plt.savefig('{}/train_cnf_matrix_normalized.jpg'.format(config['analysis_path']))
    plt.figure()
    probabilities = pipeline.predict_proba(feature_matrix)
    cnf_matrix = probability_confusion_matrix(
        y_true, y_pred, probabilities, FungusDataset.NUMBER_TO_FUNGUS)
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS,
                    'Train probability cnf matrix')
    plt.savefig('{}/train_probability_cnf_matrix.jpg'.format(config['analysis_path']))

    # test
    feature_matrix = np.load('{}/test_feature_matrix.npy'.format(config['analysis_path']))
    y_true = np.load('{}/test_labels.npy'.format(config['analysis_path']))
    y_pred = pipeline.predict(feature_matrix)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure()
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS,
                    'Test cnf matrix')
    plt.savefig('{}/test_cnf_matrix.jpg'.format(config['analysis_path']))
    plt.figure()
    plot_accuracy_bars(
        cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS, 'Test accuracy')
    plt.savefig('{}/test_accuracy_bars.jpg'.format(config['analysis_path']))
    plt.figure()
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS,
                    'Test normalized cnf matrix', normalize=True)
    plt.savefig('{}/test_cnf_matrix_normalized.jpg'.format(config['analysis_path']))
    plt.figure()
    probabilities = pipeline.predict_proba(feature_matrix)
    cnf_matrix = probability_confusion_matrix(
        y_true, y_pred, probabilities, FungusDataset.NUMBER_TO_FUNGUS)
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS,
                    'Test probability cnf matrix')
    plt.savefig('{}/test_probability_cnf_matrix.jpg'.format(config['analysis_path']))
