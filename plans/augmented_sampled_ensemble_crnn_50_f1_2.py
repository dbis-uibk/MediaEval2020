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

dataloader = MelSpectPickleLoader(
    'data/mediaeval2020/melspect_augmented_1366_sampled.pickle')

label_splits = [
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'sexy',
            'retro',
            'hopeful',
            'travel',
            'soundscape',
            'cool',
            'space',
            'holiday',
            'game',
            'nature',
            'movie',
            'background',
            'groovy',
            'fast',
            'sport',
            'heavy',
            'action',
            'drama',
            'slow',
            'powerful',
            'uplifting',
            'calm',
            'funny',
            'dramatic',
            'fun',
            'adventure',
            'trailer',
            'melancholic',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'soft',
            'party',
            'documentary',
            'ballad',
            'positive',
            'upbeat',
            'inspiring',
            'romantic',
            'melodic',
            'commercial',
            'children',
            'dream',
            'love',
            'advertising',
            'christmas',
            'corporate',
            'meditative',
            'sad',
            'motivational',
            'energetic',
            'emotional',
            'relaxing',
            'summer',
            'dark',
            'happy',
            'epic',
            'film',
            'deep',
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
