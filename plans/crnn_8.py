from dbispipeline.evaluators import EpochEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
from sklearn.pipeline import Pipeline

from mediaeval2020 import common
from mediaeval2020.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2020.models.crnn import CRNNModel

dataloader = MelSpectPickleLoader('data/mediaeval2020/melspect_1366.pickle')

pipeline = Pipeline([
    ("model", CRNNModel(epochs=8, dataloader=dataloader)),
])

evaluator = ModelCallbackWrapper(
    EpochEvaluator(**common.fixed_split_params(), scoring_step_size=5),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
