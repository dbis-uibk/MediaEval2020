"""Ensemble plan split by pr-auc based on cnn results."""
from dbispipeline.evaluators import FixedSplitEvaluator
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
            'sexy',
            'cool',
            'fast',
            'groovy',
            'retro',
            'movie',
            'drama',
            'action',
            'space',
            'hopeful',
            'holiday',
            'soundscape',
            'game',
            'dramatic',
            'nature',
            'background',
            'ballad',
            'powerful',
            'slow',
            'funny',
            'travel',
            'upbeat',
            'soft',
            'melancholic',
            'adventure',
            'fun',
            'calm',
            'uplifting',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[
            'documentary',
            'heavy',
            'inspiring',
            'party',
            'romantic',
            'positive',
            'sport',
            'commercial',
            'dream',
            'advertising',
            'sad',
            'love',
            'trailer',
            'christmas',
            'melodic',
            'meditative',
            'motivational',
            'relaxing',
            'corporate',
            'energetic',
            'emotional',
            'children',
            'dark',
            'happy',
            'epic',
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
