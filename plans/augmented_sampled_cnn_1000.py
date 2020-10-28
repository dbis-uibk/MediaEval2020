"""CNN plan with 1000 epochs."""
from dbispipeline.evaluators import EpochEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
from sklearn.pipeline import Pipeline

from mediaeval2020 import common
from mediaeval2020.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2020.models.cnn import CNNModel

dataloader = MelSpectPickleLoader(
    'data/mediaeval2020/melspect_augmented_1366_sampled.pickle')

pipeline = Pipeline([
    (
        'model',
        CNNModel(
            epochs=1000,
            dataloader=dataloader,
            block_sizes=[
                32,
                32,
                64,
                64,
                64,
            ],
        ),
    ),
])

evaluator = ModelCallbackWrapper(
    EpochEvaluator(**common.fixed_split_params(), scoring_step_size=2),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
