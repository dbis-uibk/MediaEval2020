"""Ensemble plan split by pr-auc based on cnn results."""
from dbispipeline.evaluators import FixedSplitEvaluator
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
            'groovy',
            'travel',
            'hopeful',
            'cool',
            'sexy',
            'holiday',
            'retro',
            'background',
            'game',
            'fast',
            'sport',
            'movie',
            'funny',
            'drama',
            'action',
            'fun',
            'party',
            'space',
            'slow',
            'nature',
            'melancholic',
            'powerful',
            'calm',
            'dramatic',
            'upbeat',
            'soundscape',
            'uplifting',
            'soft',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'documentary',
            'ballad',
            'positive',
            'commercial',
            'adventure',
            'romantic',
            'inspiring',
            'trailer',
            'advertising',
            'motivational',
            'corporate',
            'melodic',
            'love',
            'dream',
            'sad',
            'children',
            'meditative',
            'summer',
            'christmas',
            'relaxing',
            'energetic',
            'heavy',
            'happy',
            'emotional',
            'dark',
            'deep',
            'epic',
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
