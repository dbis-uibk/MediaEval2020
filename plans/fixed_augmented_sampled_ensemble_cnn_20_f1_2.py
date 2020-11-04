"""Ensemble plan manually split by type moode/theme."""
from dbispipeline.evaluators import FixedSplitEvaluator
from dbispipeline.evaluators import ModelCallbackWrapper
import dbispipeline.result_handlers
from sklearn.pipeline import Pipeline

from mediaeval2020 import common
from mediaeval2020.dataloaders.melspectrograms import MelSpectPickleLoader
from mediaeval2020.dataloaders.melspectrograms import labels_to_indices
from mediaeval2020.models.cnn import CNNModel
from mediaeval2020.models.ensemble import Ensemble

dataloader = MelSpectPickleLoader(
    'data/mediaeval2020/melspect_augmented_1366_sampled.pickle')

label_splits = [
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'sexy',
            'travel',
            'retro',
            'soundscape',
            'heavy',
            'fast',
            'hopeful',
            'holiday',
            'cool',
            'groovy',
            'background',
            'party',
            'drama',
            'slow',
            'nature',
            'movie',
            'game',
            'ballad',
            'powerful',
            'sport',
            'space',
            'action',
            'trailer',
            'calm',
            'funny',
            'dramatic',
            'upbeat',
            'adventure',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'melancholic',
            'uplifting',
            'fun',
            'positive',
            'soft',
            'documentary',
            'inspiring',
            'romantic',
            'commercial',
            'love',
            'corporate',
            'meditative',
            'advertising',
            'dream',
            'melodic',
            'sad',
            'christmas',
            'children',
            'deep',
            'motivational',
            'dark',
            'energetic',
            'relaxing',
            'epic',
            'happy',
            'summer',
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
         epochs=20,
     )),
])

evaluator = ModelCallbackWrapper(
    FixedSplitEvaluator(**common.fixed_split_params()),
    lambda model: common.store_prediction(model, dataloader),
)

result_handlers = [
    dbispipeline.result_handlers.print_gridsearch_results,
]
