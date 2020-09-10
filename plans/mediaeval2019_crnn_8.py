from dbispipeline.evaluators import EpochEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
from sklearn.pipeline import Pipeline

from mediaeval2020 import common
from mediaeval2020.dataloaders.melspectrograms import MelSpectrogramsLoader
from mediaeval2020.models.crnn import CRNNModel

WINDOW_SIZE = 1366

dataloader = MelSpectrogramsLoader(
    data_path="data/mediaeval2019/melspec_data",
    training_path="data/mediaeval2019/autotagging_moodtheme-train.tsv",
    test_path="data/mediaeval2019/autotagging_moodtheme-test.tsv",
    validate_path="data/mediaeval2019/autotagging_moodtheme-validation.tsv",
    window_size=WINDOW_SIZE,
)

pipeline = Pipeline([
    ("model", CRNNModel(epochs=8, dataloader=dataloader)),
])

evaluator = ModelCallbackWrapper(
    EpochEvaluator(**common.fixed_split_params()),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
