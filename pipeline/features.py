"""
Functions used to extract features from fungus images.
"""
import logging as log

import numpy as np
import torch
from torchvision import models
from tqdm import tqdm


def get_resnet18():
    resnet = models.resnet.resnet18(pretrained=True)
    features = torch.nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
        resnet.avgpool
    )
    return features


def get_inception_v3():
    inceptionv3 = models.inception.inception_v3(pretrained=True)
    features = torch.nn.Sequential(
        inceptionv3.Conv2d_1a_3x3,
        inceptionv3.Conv2d_2a_3x3,
        inceptionv3.Conv2d_2b_3x3,
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        inceptionv3.Conv2d_3b_1x1,
        inceptionv3.Conv2d_4a_3x3,
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        inceptionv3.Mixed_5b,
        inceptionv3.Mixed_5c,
        inceptionv3.Mixed_5d,
        inceptionv3.Mixed_6a,
        inceptionv3.Mixed_6b,
        inceptionv3.Mixed_6c,
        inceptionv3.Mixed_6d,
        inceptionv3.Mixed_6e,
        inceptionv3.Mixed_7a,
        inceptionv3.Mixed_7b,
        inceptionv3.Mixed_7c,
        torch.nn.AvgPool2d(kernel_size=8),
        torch.nn.Dropout(),
    )
    return features


extractors = dict()

extractors['alexnet'] = models.alexnet(pretrained=True).features
extractors['resnet18'] = get_resnet18()
extractors['inceptionv3'] = get_inception_v3()


def get_cuda():
    """Return cuda device if available else return cpu."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def extract_features(images, device, extractor):
    """Extract features from a set of images

    images -- set of images with shape N, C, W, H
    extractor -- pytorch model to be used as an extractor. If None then
    AlexNet will be used.

    Returns 3D tensor (N, W * H, C).
    """
    features = extractor(images.float())
    N, C, W, H = features.size()
    return features.reshape(N, C, W * H).transpose_(2, 1)


def compute_feature_matrix(loader, device, extractor='alexnet'):
    """Compute feature matrix from entire dataset provided by loader.

    loader - pytorch DataLoader used to draw samples
    extractor - pytorch model used to extract features, if None then AlexNet will be used
    """
    with torch.no_grad():
        extractor = extractors[extractor].eval().to(device)
        # needs to be done on cpu, out-of-memory otherwise
        image_patches = torch.tensor(
            [], dtype=torch.float, device=torch.device('cpu'))
        feature_matrix = torch.tensor([], dtype=torch.float, device=device)
        labels = torch.tensor([], dtype=torch.long)
        for i, sample in enumerate(tqdm(loader)):
            image_patches = torch.cat(
                (image_patches, sample['image'].float()), dim=0)
            X = sample['image'].to(device)
            y_true = sample['class']
            X_features = extract_features(X, device, extractor)
            feature_matrix = torch.cat((feature_matrix, X_features), dim=0)
            labels = torch.cat((labels, y_true), dim=0)
    return image_patches.numpy(), feature_matrix.cpu().numpy(), labels.numpy()
