#!/usr/bin/env python
import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

from dataset import FungusDataset
from util.config import load_config
from util.log import get_logger
from util.log import set_excepthook
from util.path import get_results_path
from util.random import set_seed

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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', help='input file prefix')
    parser.add_argument('--bow', default=False,
                        action='store_true', help='enable bow pipeline')
    parser.add_argument('--config', default='experiments_config.py',
                        help='path to python module with shared experiment configuration')
    return parser.parse_args()


def plot_all(path, mode, cnf_matrix, proba_cnf_matrix):
    plot_accuracy_bars(cnf_matrix,
                       FungusDataset.NUMBER_TO_FUNGUS,
                       '{} accuracy'.format(mode),
                       path / 'accuracy_bars.png')
    plot_cnf_matrix(cnf_matrix,
                    FungusDataset.NUMBER_TO_FUNGUS,
                    '{} confusion matrix'.format(mode),
                    path / 'confusion_matrix.png')
    plot_cnf_matrix(cnf_matrix,
                    FungusDataset.NUMBER_TO_FUNGUS,
                    '{} normalized confusion matrix'.format(mode),
                    path / 'normalized_confusion_matrix.png',
                    normalize=True)
    plot_cnf_matrix(proba_cnf_matrix,
                    FungusDataset.NUMBER_TO_FUNGUS,
                    '{} probability confusion matrix'.format(mode),
                    path / 'probability_confusion_matrix.png')


def process(features_path, model_path, results_path, mode):
    pipeline = joblib.load(model_path / 'best_model.pkl')
    feature_matrix = np.load(features_path / 'feature_matrix.npy')
    y_true = np.load(features_path / 'labels.npy')

    y_pred = pipeline.predict(feature_matrix)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    probabilities = pipeline.predict_proba(feature_matrix)
    proba_cnf_matrix = probability_confusion_matrix(
        y_true, y_pred, probabilities, FungusDataset.NUMBER_TO_FUNGUS)

    results_path.mkdir(parents=True, exist_ok=True)
    plot_all(results_path, mode, cnf_matrix, proba_cnf_matrix)


if __name__ == '__main__':
    logger = get_logger('confusion_matrixes')
    set_excepthook(logger)

    args = parse_arguments()
    config = load_config(args.config)
    set_seed(config.seed)
    model = 'bow' if args.bow else 'fv'
    features_path = get_results_path(
        config.results_path, 'features', args.prefix, 'train')
    train_results_path = get_results_path(
        config.results_path, model, args.prefix, 'train')
    test_results_path = get_results_path(
        config.results_path, model, args.prefix, 'test')
    train_features_path = get_results_path(
        config.results_path, 'features', args.prefix, 'train')
    test_features_path = get_results_path(
        config.results_path, 'features', args.prefix, 'test')
    logger.info('Plotting charts for prefix %s with %s model',
                args.prefix, model)
    process(train_features_path, train_results_path,
            train_results_path, 'train')
    process(test_features_path, train_results_path, test_results_path, 'test')
    logger.info('Plotting successfull')
