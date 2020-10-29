"""Ensemble plan manually split by type moode/theme."""
from dbispipeline.evaluators import EpochEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
from sklearn.pipeline import Pipeline

from mediaeval2020 import common
from mediaeval2020.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2020.dataloaders.melspectrograms import labels_to_indices
from mediaeval2020.models.crnn import CRNNModel
from mediaeval2020.models.ensemble import Ensemble

dataloader = MelSpectPickleLoader('data/mediaeval2020/melspect_1366.pickle')

label_splits = [
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'travel',
            'sexy',
            'retro',
            'groovy',
            'fast',
            'cool',
            'holiday',
            'soundscape',
            'hopeful',
            'game',
            'space',
            'action',
            'dramatic',
            'background',
            'heavy',
            'movie',
            'nature',
            'drama',
            'slow',
            'sport',
            'funny',
            'party',
            'calm',
            'trailer',
            'ballad',
            'melancholic',
            'fun',
            'adventure',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'positive',
            'upbeat',
            'powerful',
            'soft',
            'inspiring',
            'documentary',
            'uplifting',
            'romantic',
            'commercial',
            'corporate',
            'christmas',
            'advertising',
            'dream',
            'motivational',
            'children',
            'melodic',
            'sad',
            'meditative',
            'love',
            'summer',
            'energetic',
            'relaxing',
            'dark',
            'deep',
            'epic',
            'emotional',
            'happy',
            'film',
        ],
    ),
]

pipeline = Pipeline([
    ('model',
     Ensemble(
         base_estimator=CRNNModel(dataloader=dataloader),
         label_splits=label_splits,
         epochs=50,
     )),
])

evaluator = ModelCallbackWrapper(
    EpochEvaluator(**common.fixed_split_params(), scoring_step_size=2),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
