from sklearn import ensemble
from sklearn import svm
from sklearn.pipeline import Pipeline

from pipeline.bow import BagOfWordsTransformer
from pipeline.fisher_vector_transformer import FisherVectorTransformer

pipelines = dict()

pipelines['fv_svc'] = Pipeline(
    steps=[
        ('fisher_vector', FisherVectorTransformer()),
        ('svc', svm.SVC(probability=True)),
    ]
)

pipelines['bow_svc'] = Pipeline(
    steps=[
        ('bag_of_words', BagOfWordsTransformer()),
        ('svc', svm.SVC(probability=True)),
    ]
)

pipelines['fv_rf'] = Pipeline(
    steps=[
        ('fisher_vector', FisherVectorTransformer()),
        ('rf', ensemble.RandomForestClassifier()),
    ]
)

pipelines['bow_rf'] = Pipeline(
    steps=[
        ('bag_of_words', BagOfWordsTransformer()),
        ('rf', ensemble.RandomForestClassifier()),
    ]
)
