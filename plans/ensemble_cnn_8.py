"""One vs Rest CNN plan."""
from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
import numpy as np
from sklearn.pipeline import Pipeline

from mediaeval2020 import common
from mediaeval2020.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2020.models.cnn import CNNModel
from mediaeval2020.models.ensemble import Ensemble

dataloader = MelSpectPickleLoader('data/mediaeval2020/melspect_1366.pickle')

label_splits = np.array([
    np.arange(0, 14, 1),
    np.arange(14, 28, 1),
    np.arange(28, 42, 1),
    np.arange(42, 56, 1),
])

pipeline = Pipeline([
    ('model',
     Ensemble(
         base_estimator=CNNModel(
             epochs=8,
             dataloader=dataloader,
             block_sizes=[
                 32,
                 32,
                 64,
                 64,
                 64,
             ],
         ),
         label_splits=label_splits,
     )),
])

evaluator = ModelCallbackWrapper(
    FixedSplitEvaluator(**common.fixed_split_params()),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
