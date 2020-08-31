"""Scikit learn diabetes dataset example dataloader."""

from dbispipeline.base import Loader
import numpy as np
from sklearn import datasets


class DiabetesLoader(Loader):
    """Loads the diabetes dataset from sklearn."""

    def __init__(self):
        """Intitializes the dataloader object."""
        pass

    def load(self):
        """Returns the data."""
        diabetes = datasets.load_diabetes()

        # Use only one feature
        data = diabetes.data[:, np.newaxis, 2]
        target = diabetes.target
        return data, target

    @property
    def configuration(self):
        """Returns a dict-like representation of the configuration."""
        return {
            'name': 'DiabetesLoader',
        }
