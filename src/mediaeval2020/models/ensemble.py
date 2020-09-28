"""Module contains ensemble models."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone


class Ensemble(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator, label_splits):
        self.base_estimator = base_estimator
        self.label_splits = label_splits

        self.models = []

    def fit(self, data, labels, epochs=None):
        if len(self.models) == 0:
            for split in self.label_splits:
                split_model = clone(self.base_estimator)
                split_model.label_split = split
                self.models.append(split_model)

        for model, split in zip(self.models, self.label_splits):
            model.fit(data, labels[..., split], epochs=epochs)

    def predict(self, data):
        return self._ensamble_predict(data, prediction_type='label')

    def predict_proba(self, data):
        return self._ensamble_predict(data, prediction_type='proba')

    def _ensamble_predict(self, data, prediction_type):
        predictions = []
        for model in self.models:
            if prediction_type == 'proba':
                predictions.append(model.predict_proba(data))
            elif prediction_type == 'label':
                predictions.append(model.predict(data))

        predictions = np.stack(predictions, axis=-1)
        return predictions[..., np.argsort(self.label_splits.ravel())]
