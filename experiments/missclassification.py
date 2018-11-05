#!/usr/bin/env python
import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import argparse
import logging as log

import numpy as np
import torch
from skimage import io
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from torch.utils import data
from tqdm import tqdm

from config import config
from DataLoader import FungusDataset
from pipeline import features
from pipeline.classification import FisherVectorTransformer

if __name__ == '__main__':
    pipeline = Pipeline(
        steps=[
            ('fisher_vector', FisherVectorTransformer()),
            ('svc', svm.SVC())
        ]
    )
    pipeline = joblib.load('results/2_3_250_50p_best_model.pkl')
    device = features.get_cuda()
    dataset = FungusDataset(
        pngs_dir='/home/arccha/fungus_data_png/pngs_50p/',
        masks_dir='/home/arccha/fungus_data_png/masks_2_3_50p/',
        random_crop_size=250,
        number_of_bg_slices_per_image=1,
        number_of_fg_slices_per_image=16,
        train=False)
    loader = data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=1,
        pin_memory=True)

    correct = [[] for _ in range(10)]
    miss = [[] for _ in range(10)]
    miss_labels = [[] for _ in range(10)]

    with torch.no_grad():
        for i, sample in enumerate(tqdm(loader)):
            X = sample['image'].to(device)
            print(X.shape)
            y_true = sample['class'].cpu().numpy()
            X_features = features.extract_features(
                X, device, None).cpu().numpy()
            y_pred = pipeline.predict(X_features)
            for i in range(len(y_true)):
                if y_true[i] == y_pred[i]:
                    if len(correct[y_true[i]]) < 3:
                        correct[y_true[i]].append(X[i].cpu().numpy())
                    else:
                        for j, img in enumerate(correct[y_true[i]]):
                            img = np.clip(img, -1, 1)
                            io.imsave(
                                str(y_true[i]) + '_' + str(j) + '.jpg', np.moveaxis(img, 0, -1))
                else:
                    if len(miss[y_true[i]]) < 1:
                        miss[y_true[i]].append(X[i].cpu().numpy())
                        miss_labels[y_true[i]].append(y_pred[i])
                    else:
                        for j, img in enumerate(miss[y_true[i]]):
                            img = np.clip(img, -1, 1)
                            io.imsave(str(y_true[i]) + '_miss_' + str(
                                miss_labels[y_true[i]][j]) + '_' + str(j) + '.jpg', np.moveaxis(img, 0, -1))
                            print('kurwa')
