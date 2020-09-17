from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from mediaeval2020 import common
from mediaeval2020.dataloaders.melspectrograms import MelSpectrogramsLoader

WINDOW_SIZE = 1366

dataloader = MelSpectrogramsLoader(
    data_path="data/mediaeval2019/melspec_data",
    training_path="data/mediaeval2019/autotagging_moodtheme-train.tsv",
    test_path="data/mediaeval2019/autotagging_moodtheme-test.tsv",
    validate_path="data/mediaeval2019/autotagging_moodtheme-validation.tsv",
    window_size=WINDOW_SIZE,
    flatten=True,
)

pipeline = Pipeline([
    ("model",
     OneVsRestClassifier(
         ExtraTreesClassifier(n_estimators=100),
         n_jobs=-1,
     )),
])

evaluator = ModelCallbackWrapper(
    FixedSplitEvaluator(**common.fixed_split_params()),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
