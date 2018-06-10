from torch import nn, Tensor
from torchvision import models
import numpy as np
from cyvlfeat.gmm import gmm
from sklearn.svm import SVC
from skimage import io
from glob import glob
import pandas as pd


def model_init():
    model = models.alexnet(pretrained=True)
    # new_classifier = nn.Sequential(*list(model.features()))
    # model.classifier = new_classifier
    return model


def image_read(path):
    image = io.imread(path)
    image = np.reshape(image, (1, image.shape[2], image.shape[0], image.shape[1]))
    image_tensor = Tensor(np.asarray(image / 256).astype(np.uint8))
    return image_tensor


def extract_features(image_tensor, model):
    data = model.features(image_tensor)
    del model, image_tensor
    data = data.detach().numpy()[0]
    data = data.reshape(data.shape[0], data.shape[2] * data.shape[1])
    data = gmm(data.transpose())
    return data


def prepare_one_image_to_classify(path):
    image_tensor = image_read(path)
    model = model_init()
    data = extract_features(image_tensor, model)
    return data

paths = glob('/home/dawid_rymarczyk/Pobrane/fungus/*/*')
paths_for_train = []
paths_for_test = []
for path in paths:
    if int(path.split('/')[-1][2:-4]) > 9:
        paths_for_test.append(path)
    else:
        paths_for_train.append(path)

if __name__ == '__main__':
    for path in paths_for_train:
        features_for_svm = prepare_one_image_to_classify(path)
        np.save(path.split('/')[-1][:-4], features_for_svm)
