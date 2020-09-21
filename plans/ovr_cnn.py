from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from mediaeval2020 import common
from mediaeval2020.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2020.models.cnn import CNNModel

dataloader = MelSpectPickleLoader('data/mediaeval2020/melspect_1366.pickle')

pipeline = Pipeline([
    ("model", OneVsRestClassifier(CNNModel(epochs=8, dataloader=dataloader))),
])

evaluator = ModelCallbackWrapper(
    FixedSplitEvaluator(**common.fixed_split_params()),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
