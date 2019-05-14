from sklearn import svm
from sklearn.pipeline import Pipeline

from pipeline.bow import BOWPooling
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


fv_pipeline = Pipeline(
    steps=[
        ('fisher_vector', FisherVectorTransformer()),
        ('svc', svm.SVC(probability=True)),
    ]
)

bow_pipeline = Pipeline(
    steps=[
        ('bag_of_words', BOWPooling()),
        ('svc', svm.SVC(probability=True)),
    ]
)

nn_models = {
    'alexnet': (init_alexnet(), False),
}
