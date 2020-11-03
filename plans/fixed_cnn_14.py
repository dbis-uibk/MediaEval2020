"""CNN plan with best number of epochs."""
from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
from sklearn.pipeline import Pipeline

from mediaeval2020 import common
from mediaeval2020.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2020.models.cnn import CNNModel

dataloader = MelSpectPickleLoader('data/mediaeval2020/melspect_1366.pickle')

pipeline = Pipeline([
    (
        'model',
        CNNModel(
            epochs=14,
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
    FixedSplitEvaluator(**common.fixed_split_params()),
    lambda model: common.store_prediction(model, dataloader, 'final_'),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
