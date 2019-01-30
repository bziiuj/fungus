import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

from dataset import FungusDataset

import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip


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


def generate_charts(mode, results_dir, prefix, bow):
    filename_prefix = '{}{}/{}_{}'.format(results_dir, prefix, mode, prefix)

    # Prepare data
    feature_matrix = np.load('{}_{}'.format(
        filename_prefix, 'feature_matrix.npy'))
    y_true = np.load('{}_{}'.format(filename_prefix, 'labels.npy'))

    y_pred = pipeline.predict(feature_matrix)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    probabilities = pipeline.predict_proba(feature_matrix)
    proba_cnf_matrix = probability_confusion_matrix(
        y_true, y_pred, probabilities, FungusDataset.NUMBER_TO_FUNGUS)

    bow_type = 'fv' if not bow else 'bow'

    # Plot charts
    plot_cnf_matrix(cnf_matrix,
                    FungusDataset.NUMBER_TO_FUNGUS,
                    'confusion matrix ({})'.format(mode),
                    '{}_{}_{}'.format(filename_prefix, bow_type, 'confusion_matrix.png'))
    plot_accuracy_bars(cnf_matrix,
                       FungusDataset.NUMBER_TO_FUNGUS,
                       'accuracy ({})'.format(mode),
                       '{}_{}_{}'.format(filename_prefix, bow_type, 'accuracy_bars.png'))
    plot_cnf_matrix(cnf_matrix,
                    FungusDataset.NUMBER_TO_FUNGUS,
                    'normalized confusion matrix ({})'.format(mode),
                    '{}_{}_{}'.format(filename_prefix, bow_type,
                                      'normalized_confusion_matrix.png'),
                    normalize=True)
    plot_cnf_matrix(proba_cnf_matrix,
                    FungusDataset.NUMBER_TO_FUNGUS,
                    'probability confusion matrix ({})'.format(mode),
                    '{}_{}_{}'.format(filename_prefix, bow_type, 'probability_confusion_matrix.png'))


if __name__ == '__main__':
    SEED = 9001
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results_dir', help='absolute path to results directory')
    parser.add_argument('--prefix', help='input file prefix')
    parser.add_argument('--bow', default=False,
                        action='store_true', help='enable bow pipeline')
    args = parser.parse_args()

    model_filename = '{}{}_{}/best_model.pkl'.format(
        args.results_dir,
        'fv' if not args.bow else 'bow',
        args.prefix,
    )
    pipeline = joblib.load(model_filename)

    generate_charts('train', args.results_dir, args.prefix, args.bow)
    generate_charts('test', args.results_dir, args.prefix, args.bow)
