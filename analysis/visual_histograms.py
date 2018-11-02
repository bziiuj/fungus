#!/usr/bin/env python

import os  # isort:skip
import sys  # isort:skip

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip

import logging as log

import matplotlib
import numpy as np
import scipy.io as sio
from DataLoader import FungusDataset
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import svm

from pipeline.classification import FisherVectorTransformer  # isort:skip

plt.switch_backend('agg')


if __name__ == '__main__':
    fv = FisherVectorTransformer(gmm_samples_number=5000)
    svc = svm.SVC(C=10.0, kernel='linear')

    # load train and test data
    train_image_patches = np.load('results/train_image_patches.npy')
    train_feature_matrix = np.load('results/train_feature_matrix.npy')
    train_labels = np.load('results/train_labels.npy')
    test_image_patches = np.load('results/test_image_patches.npy')
    test_feature_matrix = np.load('results/test_feature_matrix.npy')
    test_labels = np.load('results/test_labels.npy')

    # fit gmm with train data
    fv.fit(train_feature_matrix)

    # process train and test data with gmm
    train_fv_matrix = fv.transform(train_feature_matrix)
    test_fv_matrix = fv.transform(test_feature_matrix)

    # compute distances from train and test to gmm clusters
    train_distances = cdist(
        train_feature_matrix.reshape(-1, 256), fv.gmm_[0].transpose())
    test_distances = cdist(
        test_feature_matrix.reshape(-1, 256), fv.gmm_[0].transpose())

    # get n_samples patches closest to gmm clusters (together with precise location of the closest fragment)
    n_samples = 7
    train_cluster_patches = []
    train_cluster_patches_locations = []
    for d in range(train_distances.shape[1]):
        dist_ = train_distances[:, d]
        order_ = np.argsort(dist_)
        # take three closest patches
        three_ = order_[:n_samples] // train_feature_matrix.shape[1]
        train_cluster_patches.append(train_image_patches[three_, :, :, :])
        train_cluster_patches_locations.append(
            order_[:n_samples] % train_feature_matrix.shape[1])
    train_cluster_patches = np.stack(train_cluster_patches)
    train_cluster_patches_locations = np.stack(train_cluster_patches_locations)

    # generate train bow
    train_bows = []
    for d in range(train_feature_matrix.shape[0]):
        dist_ = cdist(train_feature_matrix[d, :], fv.gmm_[0].transpose())
        train_bows.append(np.histogram(dist_.argmin(axis=1),
                                       bins=np.arange(train_distances.shape[1] + 1))[0])
    train_bows = np.stack(train_bows)
    # sio.savemat('results/train_bow.mat', {'cluster_patches': train_cluster_patches,
    # 'labels': train_labels,
    # 'bows': train_bows,
    # 'train_cluster_patches_locations': train_cluster_patches_locations})
    #
    # generate test bow
    test_bows = []
    for d in range(test_feature_matrix.shape[0]):
        dist_ = cdist(test_feature_matrix[d, :], fv.gmm_[0].transpose())
        test_bows.append(np.histogram(dist_.argmin(axis=1),
                                      bins=np.arange(test_distances.shape[1] + 1))[0])
    test_bows = np.stack(test_bows)
    # sio.savemat('results/test_bow.mat', {'cluster_patches': train_cluster_patches,
    # 'labels': test_labels,
    # 'bows': test_bows,
    # 'train_cluster_patches_locations': train_cluster_patches_locations})
    #
    # fit classifier check accuracy
    svc.fit(train_fv_matrix, train_labels)
    log.info('Accuracy training {}'.format(
        svc.score(train_fv_matrix, train_labels)))
    log.info('Accuracy test {}'.format(svc.score(test_fv_matrix, test_labels)))


dim = train_cluster_patches.shape[3]
step = dim // 5 - 1
print(train_cluster_patches.shape)
print(step)
print(range(dim, step))
x, y = np.meshgrid(range(dim, step), range(dim, step))
print(x)
print(y)
x = x[:]
y = y[:]
print(x)
print(y)

# TODO in matlab the `c` below is then shadowed by for loop, is this intentional?
# cluster_count, image_count, c, h, w = train_cluster_patches.shape
# plt.figure(dpi=300)
# for c in range(cluster_count):
#     for i in range(image_count):
#         # in matlab mean is computed along first dimension which size is not equal to 1
#         # as this is invoked after squeeze then it should be computed along 0 axis
#         means = np.mean(np.squeeze(train_cluster_patches[c, i, :, :, :]), axis=0)
#         patch = np.squeeze(means)
#         plt.subplot(image_count, cluster_count, i * cluster_count + c + 1)
#         indicators = train_cluster_patches_locations[c, i]
#         # should this be y, x or x, y?
#         plt.plot(y[indicators], x[indicators], 'ro')
#         plt.axis('off')
#         plt.imshow(patch)
# plt.savefig('test.png')
# plt.close()

# box plots


def plot_boxplot(bows, labels, name):
    flierprops = dict(marker='+', markerfacecolor='red')
    plt.figure(figsize=(10, 10), dpi=300)
    for i in range(10):
        i_bows = bows[labels == i, :]
        plt.subplot(2, 5, i + 1)
        plt.boxplot(i_bows, flierprops=flierprops)
        plt.title(FungusDataset.NUMBER_TO_FUNGUS[i])
    plt.savefig(name)
    plt.close()


plot_boxplot(train_bows, train_labels, 'train_boxplot.png')
plot_boxplot(test_bows, test_labels, 'test_boxplot.png')
