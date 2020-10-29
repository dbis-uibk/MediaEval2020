"""Ensemble plan manually split by type moode/theme."""
from dbispipeline.evaluators import EpochEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
from sklearn.pipeline import Pipeline

from mediaeval2020 import common
from mediaeval2020.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2020.dataloaders.melspectrograms import labels_to_indices
from mediaeval2020.models.cnn import CNNModel
from mediaeval2020.models.ensemble import Ensemble

dataloader = MelSpectPickleLoader('data/mediaeval2020/melspect_1366.pickle')

label_splits = [
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'retro',
            'sexy',
            'travel',
            'fast',
            'groovy',
            'cool',
            'holiday',
            'hopeful',
            'movie',
            'action',
            'game',
            'soundscape',
            'sport',
            'space',
            'nature',
            'background',
            'heavy',
            'ballad',
            'funny',
            'fun',
            'drama',
            'party',
            'powerful',
            'slow',
            'calm',
            'dramatic',
            'uplifting',
            'melancholic',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'upbeat',
            'positive',
            'trailer',
            'melodic',
            'adventure',
            'soft',
            'inspiring',
            'advertising',
            'commercial',
            'documentary',
            'romantic',
            'corporate',
            'motivational',
            'christmas',
            'dream',
            'love',
            'sad',
            'deep',
            'children',
            'summer',
            'meditative',
            'relaxing',
            'energetic',
            'dark',
            'happy',
            'epic',
            'emotional',
            'film',
        ],
    ),
]

pipeline = Pipeline([
    ('model',
     Ensemble(
         base_estimator=CNNModel(
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
