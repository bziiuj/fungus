#!/usr/bin/env python
import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip


import argparse
import logging as log

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from torchvision import models

from pipeline.features import get_cuda

if __name__ == '__main__':
    device = get_cuda()
    device = torch.device('cpu')
    classifier = models.alexnet(pretrained=True).classifier.eval().to(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='', help='input files prefix')
    parser.add_argument('--test', default=False,
                        action='store_true', help='enable test mode')
    args = parser.parse_args()
    if args.test:
        filename_prefix = 'results/test_'
    else:
        filename_prefix = 'results/train_'
    if args.prefix:
        filename_prefix += args.prefix
    feature_matrix_filename = filename_prefix + 'feature_matrix.npy'
    labels_filename = filename_prefix + 'labels.npy'
    feature_matrix = np.load(feature_matrix_filename)
    feature_matrix = torch.from_numpy(feature_matrix).to(device)
    feature_matrix = feature_matrix.view(feature_matrix.size(0), 256 * 6 * 6)
    labels_pred = classifier(feature_matrix)
    labels = np.load(labels_filename)
    print(labels.shape)
    print(labels_pred.size())
