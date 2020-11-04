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
            'soundscape',
            'holiday',
            'hopeful',
            'travel',
            'heavy',
            'cool',
            'movie',
            'nature',
            'game',
            'powerful',
            'action',
            'drama',
            'space',
            'fast',
            'party',
            'ballad',
            'slow',
            'retro',
            'background',
            'adventure',
            'calm',
            'melancholic',
            'dramatic',
            'groovy',
            'positive',
            'soft',
            'upbeat',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'uplifting',
            'fun',
            'documentary',
            'inspiring',
            'trailer',
            'funny',
            'romantic',
            'commercial',
            'sport',
            'love',
            'dream',
            'melodic',
            'sad',
            'advertising',
            'motivational',
            'corporate',
            'meditative',
            'energetic',
            'dark',
            'relaxing',
            'christmas',
            'children',
            'emotional',
            'happy',
            'epic',
            'summer',
            'film',
            'deep',
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
