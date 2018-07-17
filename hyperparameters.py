#!/usr/bin/env python
"""
Perform grid search for the best parameters of SVC using train dataset read from .npy files, then save the selected model to `best_model.pkl`.
"""
import numpy as np
import torch
from config import config
from pipeline import compute_fisher_vectors, fit_gmm
from sklearn import model_selection, svm
from sklearn.externals import joblib

if __name__ == '__main__':
    feature_matrix = np.load('train_feature_matrix.npy')
    labels = np.load('train_labels.npy')
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
    clf = model_selection.GridSearchCV(svm.SVC(), param_grid)
    fisher_vectors = compute_fisher_vectors(feature_matrix)
    clf.fit(fisher_vectors, labels)
    print(clf.best_params_)
    joblib.dump(clf, 'best_model.pkl')
    print(clf.cv_results_)
