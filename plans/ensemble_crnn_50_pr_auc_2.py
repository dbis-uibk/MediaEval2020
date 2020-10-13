"""Ensemble plan split by pr-auc based on cnn results."""
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
            'retro',
            'fast',
            'soundscape',
            'holiday',
            'hopeful',
            'cool',
            'groovy',
            'nature',
            'space',
            'game',
            'party',
            'slow',
            'movie',
            'drama',
            'action',
            'melancholic',
            'dramatic',
            'calm',
            'background',
            'funny',
            'positive',
            'ballad',
            'upbeat',
            'uplifting',
            'romantic',
            'adventure',
            'soft',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'fun',
            'powerful',
            'sport',
            'trailer',
            'corporate',
            'sexy',
            'commercial',
            'inspiring',
            'advertising',
            'documentary',
            'motivational',
            'melodic',
            'dream',
            'sad',
            'christmas',
            'meditative',
            'love',
            'children',
            'heavy',
            'summer',
            'relaxing',
            'energetic',
            'deep',
            'dark',
            'emotional',
            'happy',
            'epic',
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
