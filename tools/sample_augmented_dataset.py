"""Selects windows from augmented dataset."""
import pickle
import random

from logzero import logger
import numpy as np


def _log_shape(dataset, subset):
    logger.info(
        '%s subset:  data: %d, labels: %d',
        subset,
        dataset[subset][0].shape,
        dataset[subset][1].shape,
    )


def _select_samples(dataset, num_windows):
    samples = []
    labels = []
    for x, y in zip(dataset):
        if _keep_sample(y, num_windows):
            samples.append(x)
            labels.append(y)

    return np.array(samples), np.array(labels)


def _keep_sample(label, num_windows):
    for group, labels in _label_groups().items():
        if label in labels:
            return _get_proba(group, num_windows) < random.random()


def _select_test(dataset, num_windows):
    samples = []
    labels = []
    for i, (x, y) in enumerate(zip(dataset)):
        if i % 14 == 0:
            samples.append(x)
            labels.append(y)

    return np.array(samples), np.array(labels)


def _get_proba(group, num_windows):
    return group / num_windows


def _label_groups():
    return {
        1: [11, 16, 17, 18, 19, 21, 26, 30, 31, 34, 41],
        2: [2, 7, 10, 12, 13, 32, 35, 39, 43, 44, 48, 49, 55],
        3: [0, 1, 3, 5, 14, 22, 33, 46, 47, 51, 54],
        4: [
            4,
            6,
            8,
            9,
            15,
            20,
            23,
            24,
            25,
            27,
            28,
            29,
            36,
            37,
            38,
            40,
            42,
            45,
            50,
            52,
            53,
        ],
    }


file_prefix = 'data/mediaeval2020/melspect_augmented_1366'
file_name = file_prefix + '.pickle'
with open(file_name, 'rb') as f:
    data = pickle.load(f)

logger.info('Select samples for train.')
_log_shape(data, 'train')
data['train'] = _select_samples(
    data['train'],
    data['configuration']['num_windows'],
)
_log_shape(data, 'train')

logger.info('Select samples for validate.')
_log_shape(data, 'validate')
data['validate'] = _select_samples(data['validate'],
                                   data['configuration']['num_windows'])
_log_shape(data, 'validate')

logger.info('Select samples for test.')
_log_shape(data, 'test')
data['test'] = _select_test(data['test'], data['configuration']['num_windows'])
_log_shape(data, 'test')

data['configuration']['label_groups'] = _label_groups()

file_name = file_prefix + '_sampled.pickle'
logger.info('Store melspect data')
pickle.dump(
    data,
    open(file_name, 'wb'),
    protocol=pickle.HIGHEST_PROTOCOL,
)
