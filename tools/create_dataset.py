"""Creates a MediaEval2020 dataset based on MediaEval2019."""
import pickle

from logzero import logger

from mediaeval2020.dataloaders.melspectrograms import MelSpectrogramsLoader

logger.info('Load data using the MelSpectrogramLoader')
WINDOW_SIZE = 1366
dataloader = MelSpectrogramsLoader(
    data_path="data/mediaeval2019/melspec_data",
    training_path="data/mediaeval2019/autotagging_moodtheme-train.tsv",
    test_path="data/mediaeval2019/autotagging_moodtheme-test.tsv",
    validate_path="data/mediaeval2019/autotagging_moodtheme-validation.tsv",
    window_size=WINDOW_SIZE,
)

file_prefix = 'data/mediaeval2020'
file_prefix += ('/melspect_' + str(WINDOW_SIZE))

logger.info('Store train data')
pickle.dump(
    dataloader.load_train(),
    open(file_prefix + '_train.pickle', 'wb'),
)

logger.info('Store validate data')
pickle.dump(
    dataloader.load_validate(),
    open(file_prefix + '_validate.pickle', 'wb'),
)

logger.info('Store test data')
pickle.dump(
    dataloader.load_test(),
    open(file_prefix + '_test.pickle', 'wb'),
)
