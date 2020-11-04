"""Ensemble plan manually split by type moode/theme."""
from dbispipeline.evaluators import FixedSplitEvaluator
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
            'travel',
            'hopeful',
            'soundscape',
            'groovy',
            'fast',
            'action',
            'cool',
            'space',
            'movie',
            'game',
            'drama',
            'nature',
            'retro',
            'dramatic',
            'slow',
            'holiday',
            'calm',
            'background',
            'sport',
            'soft',
            'ballad',
            'adventure',
            'documentary',
            'powerful',
            'uplifting',
            'fun',
            'funny',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'party',
            'upbeat',
            'melancholic',
            'heavy',
            'romantic',
            'melodic',
            'positive',
            'dream',
            'trailer',
            'commercial',
            'inspiring',
            'advertising',
            'sad',
            'christmas',
            'meditative',
            'energetic',
            'love',
            'motivational',
            'emotional',
            'children',
            'relaxing',
            'corporate',
            'dark',
            'epic',
            'happy',
            'film',
            'summer',
            'deep',
        ],
    ),
]

pipeline = Pipeline([
    ('model',
     Ensemble(
         base_estimator=CRNNModel(dataloader=dataloader),
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
