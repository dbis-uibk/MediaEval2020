"""Module contains ensemble models."""
import numpy as np
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class Ensemble(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator, label_splits):
        self.base_estimator = base_estimator
        self.label_splits = label_splits

        self.models = []

    def fit(self, data, labels, epochs=None):
        if len(self.models) == 0:
            for split in self.label_splits:
                split_model = clone(self.base_estimator)
                self.models.append(split_model)

        for model, split in zip(self.models, self.label_splits):
            model.fit(data, labels[..., split], epochs=epochs)

    def predict(self, data):
        return self._ensamble_predict(data, type='label')

    def predict_proba(self, data):
        return self._ensamble_predict(data, type='proba')

    def _ensamble_predict(self, data, type):
        predictions = []
        for model in self.models:
            if type == 'proba':
                predictions.append(model.predict_proba(data))
            elif type == 'label':
                predictions.append(model.predict(data))

        predictions = np.array(predictions).ravel()
        return predictions[self.label_splits.ravel()]
