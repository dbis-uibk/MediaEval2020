"""Ensemble plan linear split."""
from dbispipeline.evaluators import EpochEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
import numpy as np
from sklearn.pipeline import Pipeline

from mediaeval2020 import common
from mediaeval2020.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2020.models.crnn import CRNNModel
from mediaeval2020.models.ensemble import Ensemble

dataloader = MelSpectPickleLoader('data/mediaeval2020/melspect_1366.pickle')

label_splits = [
    np.arange(0, 28, 1),
    np.arange(28, 56, 1),
]

pipeline = Pipeline([
    ('model',
     Ensemble(
         base_estimator=CRNNModel(dataloader=dataloader),
         label_splits=label_splits,
         epochs=200,
     )),
])

evaluator = ModelCallbackWrapper(
    EpochEvaluator(**common.fixed_split_params(), scoring_step_size=2),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
