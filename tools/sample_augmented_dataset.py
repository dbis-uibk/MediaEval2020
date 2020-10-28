"""Selects windows from augmented dataset."""
import pickle
import random
import sys

from logzero import logger
import numpy as np


def _log_shape(dataset, subset):
    logger.info(
        '%s subset:  data: %s, labels: %s',
        subset,
        str(dataset[subset][0].shape),
        str(dataset[subset][1].shape),
    )


def _select_samples(dataset, num_windows, boost=1):
    samples = []
    labels = []
    for x, y in zip(*dataset):
        if _keep_sample(y, num_windows, boost=boost):
            samples.append(x)
            labels.append(y)

    return np.array(samples), np.array(labels)


def _keep_sample(labels, num_windows, boost=1):
    for group in range(1, 5):
        label_group = _label_groups()[group]
        for label in label_group:
            if labels[label] == 1:
                return (_get_proba(group, num_windows) *
                        boost) >= random.random()


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


if len(sys.argv) > 1:
    supersampling = float(sys.argv[1])
else:
    supersampling = 1

file_prefix = 'data/mediaeval2020/melspect_augmented_1366'
file_name = file_prefix + '.pickle'
with open(file_name, 'rb') as f:
    data = pickle.load(f)

logger.info('Select samples for train.')
_log_shape(data, 'train')
data['train'] = _select_samples(
    data['train'],
    data['configuration']['num_windows'],
    boost=supersampling,
)
_log_shape(data, 'train')

logger.info('Select samples for validate.')
_log_shape(data, 'validate')
data['validate'] = _select_samples(
    data['validate'],
    data['configuration']['num_windows'],
    boost=supersampling,
)
_log_shape(data, 'validate')

data['configuration']['label_groups'] = _label_groups()

if supersampling == 1:
    file_name = file_prefix + '_sampled.pickle'
else:
    file_name = file_prefix + '_super' + str(supersampling) + '_sampled.pickle'

logger.info('Store melspect data')
pickle.dump(
    data,
    open(file_name, 'wb'),
    protocol=pickle.HIGHEST_PROTOCOL,
)
