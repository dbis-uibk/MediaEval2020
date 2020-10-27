"""Selects windows from augmented dataset."""
import pickle

from logzero import logger


def _log_shape(data, subset):
    logger.info(subset + ':', data[subset][0].shape, data[subset][1].shape)


file_name = 'data/mediaeval2020/melspect_augmented_1366.pickle'
with open(file_name, 'rb') as f:
    data = pickle.load(f)

_log_shape(data, 'train')
_log_shape(data, 'validate')
_log_shape(data, 'test')
