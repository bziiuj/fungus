#!/usr/bin/env python
"""
Read feature matrix and labels from .npy files and classify them. In train
mode use train dataset, fit GMM and then fit SVC, in test mode load best
model obtained from `hyperparameters.py` and perform prediction on test
dataset.
"""
import argparse
import logging

import numpy as np
import torch
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from pipeline.fisher_vector_transformer import FisherVectorTransformer

import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip


if __name__ == '__main__':
    log = logging.getLogger('fungus')
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler('tmp/fungus.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    log.addHandler(ch)
    log.addHandler(fh)

    SEED = 9001
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'results_dir', help='absolute path to results directory')
    parser.add_argument('--test', default=False,
                        action='store_true', help='enable test mode')
    parser.add_argument('--prefix', default='', help='result filenames prefix')
    parser.add_argument('--gmm-clusters-number')
    parser.add_argument('--kernel')
    parser.add_argument('--C')
    parser.add_argument('--gamma')
    args = parser.parse_args()
    filename_prefix = '{}/{}/{}'.format(args.results_dir,
                                        args.prefix, 'test' if args.test else 'train')

    feature_matrix = np.load('{}_{}'.format(
        filename_prefix, 'feature_matrix.npy'))
    labels = np.load('{}_{}'.format(filename_prefix, 'labels.npy'))

    if args.test:
        pipeline = joblib.load(
            '{}/{}_best_model.pkl'.format(args.results_dir, args.prefix))
    else:
        pipeline = Pipeline(
            steps=[
                ('fisher_vector', FisherVectorTransformer(
                    gmm_clusters_number=int(args.gmm_clusters_number))),
                ('svc', svm.SVC(C=float(args.C), kernel=args.kernel,
                                gamma='auto' if args.gamma == 'auto' else float(
                                    args.gamma),
                                probability=True))
            ]
        )
        pipeline.fit(feature_matrix, labels)
        joblib.dump(
            pipeline, '{}/{}_best_model.pkl'.format(args.results_dir, args.prefix))

    log.info('Accuracy {}: {}'.format('test' if args.test else 'train',
                                      pipeline.score(feature_matrix, labels)))
