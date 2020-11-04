"""CRNN plan."""
from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
from sklearn.pipeline import Pipeline

from mediaeval2020 import common
from mediaeval2020.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2020.models.crnn import CRNNModel

dataloader = MelSpectPickleLoader(
    'data/mediaeval2020/melspect_augmented_1366_sampled.pickle')

pipeline = Pipeline([
    ('model', CRNNModel(epochs=20, dataloader=dataloader)),
])

evaluator = ModelCallbackWrapper(
    FixedSplitEvaluator(**common.fixed_split_params()),
    lambda model: common.store_prediction(model, dataloader, 'final_'),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
