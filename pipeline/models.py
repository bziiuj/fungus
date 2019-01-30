from sklearn import svm
from sklearn.pipeline import Pipeline

from pipeline.bow import BOWPooling
from pipeline.fisher_vector_transformer import FisherVectorTransformer

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
