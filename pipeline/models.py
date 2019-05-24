from sklearn import svm
from sklearn.pipeline import Pipeline

from pipeline.bow import BagOfWordsTransformer
from pipeline.fisher_vector_transformer import FisherVectorTransformer

from torchvision import models
from torch import nn


num_classes = 10


def init_alexnet():
    m = models.alexnet(pretrained=True, progress=True)
    for param in m.parameters():
        param.requires_grad = False
    m.classifier[6] = nn.Linear(4096, num_classes)
    return m


def init_resnet18():
    m = models.resnet18(pretrained=True, progress=True)
    for param in m.parameters():
        param.requires_grad = False
    m.fc = nn.Linear(512, num_classes)
    return m


def init_resnet50():
    m = models.resnet50(pretrained=True, progress=True)
    for param in m.parameters():
        param.requires_grad = False
    m.fc = nn.Linear(2048, num_classes)
    return m


def init_densenet():
    m = models.densenet169(pretrained=True, progress=True)
    for param in m.parameters():
        param.requires_grad = False
    m.classifier = nn.Linear(1664, num_classes)
    return m


def init_inceptionv3():
    m = models.inception_v3(pretrained=True, progress=True)
    for param in m.parameters():
        param.requires_grad = False
    m.AuxLogits.fc = nn.Linear(768, num_classes)
    m.fc = nn.Linear(2048, num_classes)
    return m


fv_pipeline = Pipeline(
    steps=[
        ('fisher_vector', FisherVectorTransformer()),
        ('svc', svm.SVC(probability=True)),
    ]
)

bow_pipeline = Pipeline(
    steps=[
        ('bag_of_words', BagOfWordsTransformer()),
        ('svc', svm.SVC(probability=True)),
    ]
)

nn_models = {
    'alexnet': (init_alexnet(), False),
    'resnet18': (init_resnet18(), False),
    'resnet50': (init_resnet50(), False),
    'densenet': (init_densenet(), False),
    'inceptionv3': (init_inceptionv3(), True),
}
