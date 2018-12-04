import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import argparse
import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from DataLoader import FungusDataset
from pipeline.classification import FisherVectorTransformer

plt.switch_backend('agg')


def probability_confusion_matrix(y_true, y_pred, probabilities, classes):
    n_classes = len(classes.keys())
    dim = (n_classes, n_classes)
    matrix = np.zeros(dim)
    count_matrix = np.zeros(dim)
    for i in range(len(y_true)):
        matrix[y_true[i], y_pred[i]] += probabilities[i, y_pred[i]]
        count_matrix[y_true[i], y_pred[i]] += 1
    return np.divide(matrix, count_matrix)


def plot_cnf_matrix(matrix, classes, title, filename, normalize=False):
    plt.figure()
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
    plt.savefig(filename)


def plot_accuracy_bars(cnf_matrix, classes, title, filename):
    plt.figure()
    accuracy = np.diag(cnf_matrix) / np.sum(cnf_matrix, axis=1)
    plt.title(title)
    plt.bar(classes.values(), accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Classes')
    plt.savefig(filename)


def generate_charts(mode, filename_mask, prefix):
    # Prepare data
    feature_matrix = np.load(filename_mask.format(
        mode + '_', args.prefix, 'feature_matrix.npy'))
    y_true = np.load(filename_mask.format(
        mode + '_', args.prefix, 'labels.npy'))
    y_pred = pipeline.predict(feature_matrix)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    probabilities = pipeline.predict_proba(feature_matrix)
    proba_cnf_matrix = probability_confusion_matrix(
        y_true, y_pred, probabilities, FungusDataset.NUMBER_TO_FUNGUS)

    # Plot charts
    plot_cnf_matrix(cnf_matrix,
                    FungusDataset.NUMBER_TO_FUNGUS,
                    mode + ' cnf matrix',
                    filename_mask.format(mode, prefix, 'cnf_matrix.png'))
    plot_accuracy_bars(cnf_matrix,
                       FungusDataset.NUMBER_TO_FUNGUS,
                       mode + ' accuracy',
                       filename_mask.format(mode, prefix, 'accuracy_bars.png'))
    plot_cnf_matrix(cnf_matrix,
                    FungusDataset.NUMBER_TO_FUNGUS,
                    mode + ' normalized cnf matrix',
                    filename_mask.format(
                        mode, prefix, 'normalized_cnf_matrix.png'),
                    normalize=True)
    plot_cnf_matrix(proba_cnf_matrix,
                    FungusDataset.NUMBER_TO_FUNGUS,
                    mode + ' probability cnf matrix',
                    filename_mask.format(mode, prefix, 'probability_cnf_matrix.png'))


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='', help='input file prefix')
    args = parser.parse_args()
    filename_mask = 'results/{}{}{}'
    prefix = '' if not args.prefix else args.prefix + '_'

    pipeline = Pipeline(
        steps=[
            ('fisher_vector', FisherVectorTransformer()),
            ('svc', svm.SVC())
        ]
    )
    pipeline = joblib.load(filename_mask.format(
        '', args.prefix, 'best_model.pkl'))

    generate_charts('train', prefix, filename_mask)
    generate_charts('test', prefix, filename_mask)
