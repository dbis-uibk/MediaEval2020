"""Tool to extract labels form a dataset."""
import pickle
import sys

from logzero import logger

logger.info('Read dataset data')
with open(sys.argv[1] + '.pickle', 'rb') as f:
    dataset = pickle.load(f)

logger.info('Extract label data')
data = {
    'train': dataset['train'][1],
    'validate': dataset['validate'][1],
    'test': dataset['test'][1],
    'config': dataset['configuration'],
}

logger.info('Store label data')
pickle.dump(
    data,
    open(sys.argv[1] + '_labels.pickle', 'wb'),
    protocol=pickle.HIGHEST_PROTOCOL,
)
