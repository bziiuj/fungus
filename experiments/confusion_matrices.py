import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import argparse
import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from DataLoader import FungusDataset
from pipeline.classification import FisherVectorTransformer

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
    return np.divide(matrix, count_matrix)


def plot_accuracy_bars(cnf_matrix, classes, title):
    accuracy = np.diag(cnf_matrix) / np.sum(cnf_matrix, axis=1)
    plt.title(title)
    plt.bar(classes.values(), accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Classes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='', help='input file prefix')
    args = parser.parse_args()
    filename_prefix = 'results/{}_'
    if args.prefix:
        filename_prefix += args.prefix + '_'

    pipeline = Pipeline(
        steps=[
            ('fisher_vector', FisherVectorTransformer()),
            ('svc', svm.SVC())
        ]
    )
    model_prefix = 'results/'
    if args.prefix:
        model_prefix += args.prefix + '_'
    pipeline = joblib.load(model_prefix + 'best_model.pkl')

    # train
    train_filename_prefix = filename_prefix.format('train')
    feature_matrix = np.load(train_filename_prefix + 'feature_matrix.npy')
    y_true = np.load(train_filename_prefix + 'labels.npy')
    y_pred = pipeline.predict(feature_matrix)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure()
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS,
                    'Train cnf matrix')
    plt.savefig(train_filename_prefix + 'cnf_matrix.jpg')
    plt.figure()
    plot_accuracy_bars(
        cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS, 'Train accuracy')
    plt.savefig(train_filename_prefix + 'accuracy_bars.jpg')
    plt.figure()
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS,
                    'Train normalized cnf matrix', normalize=True)
    plt.savefig(train_filename_prefix + 'cnf_matrix_normalized.jpg')
    plt.figure()
    probabilities = pipeline.predict_proba(feature_matrix)
    cnf_matrix = probability_confusion_matrix(
        y_true, y_pred, probabilities, FungusDataset.NUMBER_TO_FUNGUS)
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS,
                    'Train probability cnf matrix')
    plt.savefig(train_filename_prefix + 'probability_cnf_matrix.jpg')

    # test
    test_filename_prefix = filename_prefix.format('test')
    feature_matrix = np.load(test_filename_prefix + 'feature_matrix.npy')
    y_true = np.load(test_filename_prefix + 'labels.npy')
    y_pred = pipeline.predict(feature_matrix)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure()
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS,
                    'Test cnf matrix')
    plt.savefig(test_filename_prefix + 'cnf_matrix.jpg')
    plt.figure()
    plot_accuracy_bars(
        cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS, 'Test accuracy')
    plt.savefig(test_filename_prefix + 'accuracy_bars.jpg')
    plt.figure()
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS,
                    'Test normalized cnf matrix', normalize=True)
    plt.savefig(test_filename_prefix + 'cnf_matrix_normalized.jpg')
    plt.figure()
    probabilities = pipeline.predict_proba(feature_matrix)
    cnf_matrix = probability_confusion_matrix(
        y_true, y_pred, probabilities, FungusDataset.NUMBER_TO_FUNGUS)
    plot_cnf_matrix(cnf_matrix, FungusDataset.NUMBER_TO_FUNGUS,
                    'Test probability cnf matrix')
    plt.savefig(test_filename_prefix + 'probability_cnf_matrix.jpg')
