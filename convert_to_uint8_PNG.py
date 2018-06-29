from skimage import io
from glob import glob
from skimage.external import tifffile
from skimage.exposure import rescale_intensity
from matplotlib import pyplot as plt

import numpy as np
import seaborn as sns
import os

paths = glob('/home/dawid_rymarczyk/fungus/*/*')
paths_for_hist = glob('fungus_stretched/*')
paths_for_train = []
paths_for_test = []
path_to_save = 'fungus_stretched/'
path_to_save_hist = 'histograms_stretched/'
for path in paths:
    img = tifffile.imread(path)
    p1, p99 = np.percentile(np.unique(img.flatten()), [1, 99])
    img = np.asarray(rescale_intensity(img, in_range=(p1, p99))).astype(float)
    img = img.reshape((img.shape[1], img.shape[2], 3)) / 2 ** 16
    # img = normalize(img)
    splitted = path.split('/')
    path_of_fung = path_to_save + splitted[-2]
    if not os.path.isdir(path_of_fung):
        os.makedirs(path_of_fung)
    io.imsave(path_of_fung + '/' + splitted[-1][:-4] + '_stretched_contrast.png', img)

for path in paths_for_hist:
    splitted = path.split('/')
    path_of_fung = path_to_save_hist + splitted[-1]
    if not os.path.isdir(path_of_fung):
        os.makedirs(path_of_fung)
    imgs = glob(path + '/*')
    hists = []
    for img in imgs:
        data = io.imread(img)
        hists.append(np.histogram(data.flatten(), bins=range(256))[0])
    hists = np.asarray(hists)
    # sns_plot = sns.tsplot(data=np.log10(np.mean(hists, axis=0)), color='red')
    # for k in range(len(hists)):
    sns.tsplot(data=np.log10(hists + 1), err_style='unit_traces', color='red')
    plt.savefig(path_of_fung + '/' + 'hist.png')
