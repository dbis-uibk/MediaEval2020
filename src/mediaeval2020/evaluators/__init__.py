"""Evaluators for mediaeval2020."""


def grid_parameters():
    """Grid parameter setup used to setup the grid search."""
    return {
        'verbose': 3,
        'cv': 5,
        'refit': False,
        'scoring': 'neg_mean_squared_error',
        'return_train_score': True,
    }
