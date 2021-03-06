"""Ensemble plan manually split by type moode/theme/unsure."""
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
        label_list=[  # unsure
            'action',
            'adventure',
            'background',
            'drama',
            'dream',
            'love',
            'melodic',
            'motivational',
            'nature',
            'party',
            'retro',
            'soundscape',
            'space',
            'sport',
            'summer',
            'upbeat',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # theme
            'advertising',
            'ballad',
            'children',
            'christmas',
            'commercial',
            'corporate',
            'documentary',
            'film',
            'game',
            'holiday',
            'movie',
            'trailer',
            'travel',
        ],
    ),
    labels_to_indices(
        dataloader=dataloader,
        label_list=[  # mood
            'calm',
            'cool',
            'dark',
            'deep',
            'dramatic',
            'emotional',
            'energetic',
            'epic',
            'fast',
            'fun',
            'funny',
            'groovy',
            'happy',
            'heavy',
            'hopeful',
            'inspiring',
            'meditative',
            'melancholic',
            'positive',
            'powerful',
            'relaxing',
            'romantic',
            'sad',
            'sexy',
            'slow',
            'soft',
            'uplifting',
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
