"""Tests the ensemble model."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import MinMaxScaler

from mediaeval2020.models.ensemble import Ensemble

data = np.arange(4 * 4).reshape((4, 4))
labels = np.arange(4 * 16).reshape((4, 16))
label_splits = np.array([
    np.array([1, 3, 5, 7]),
    np.array([0, 2, 4, 6]),
    np.array([8, 11, 14, 15]),
    np.array([13, 10, 12, 9]),
])


class DummyClassifier(BaseEstimator, ClassifierMixin):
    """Dummy estimator predicting based on the labels from fit."""

    def fit(self, data, labels, epochs=None):
        """Stores the passed args."""
        self.data_ = data
        self.labels_ = labels
        self.epochs_ = epochs

    def predict(self, data):
        """Predcits the labels passed for fit."""
        if (self.data_ == data).all():
            return self.labels_
        else:
            raise ValueError('Data is unknown, pass data to fit.')

    def predict_proba(self, data):
        """Predcits the labels from fit but min-max scalled."""
        if (self.data_ == data).all():
            return MinMaxScaler().fit_transform(self.labels_)
        else:
            raise ValueError('Data is unknown, pass data to fit.')


def test_predict():
    """Tests the predcit method."""
    ensemble = Ensemble(
        base_estimator=DummyClassifier(),
        label_splits=label_splits,
    )

    ensemble.fit(data, labels)
    predcition = ensemble.predict(data)

    assert (labels == predcition).all()


def test_predict_proba():
    """Tests the predict_proba method."""
    ensemble = Ensemble(
        base_estimator=DummyClassifier(),
        label_splits=label_splits,
    )

    ensemble.fit(data, labels, epochs=5)
    predcition = ensemble.predict_proba(data)

    assert (MinMaxScaler().fit_transform(labels) == predcition).all()
