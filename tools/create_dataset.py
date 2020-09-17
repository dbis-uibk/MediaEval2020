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

file_name = 'data/mediaeval2020'
file_name += ('/melspect_' + str(WINDOW_SIZE) + '.pickle')

logger.info('Extract data')
data = {
    'train': dataloader.load_train(),
    'test': dataloader.load_test(),
    'validate': dataloader.load_validate(),
    'configuration': dataloader.configuration,
}

logger.info('Store melspect data')
pickle.dump(
    data,
    open(file_name, 'wb'),
    protocol=pickle.HIGHEST_PROTOCOL,
)
