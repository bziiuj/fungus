#!/usr/bin/env python
import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from dataset import FungusDataset
from util.config import load_config
from util.log import get_logger
from util.log import set_excepthook
from util.path import get_results_path
from util.random import set_seed

plt.switch_backend('agg')
sns.set()


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
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(
        matrix, index=classes.values(), columns=classes.values()
    )
    fig = plt.figure()
    fmt = '.2f'
    #fmt = '.2f' if normalize else 'd'
    heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt)
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
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
    print(accuracy_score(y_true, y_pred))


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
