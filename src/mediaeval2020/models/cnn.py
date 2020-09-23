import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_curve
import tensorflow.compat.v2.keras.backend as K
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AlphaDropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model

from ..utils import cached_model_predict
from ..utils import cached_model_predict_clear
from ..utils import find_elbow


class CNNModel(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 batch_size=64,
                 epochs=10,
                 padding='same',
                 dataloader=None,
                 block_sizes=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.padding = padding
        self.dataloader = dataloader
        if block_sizes is None:
            self.block_sizes = [32, 64, 64, 64]
        else:
            self.block_sizes = block_sizes
        self.network_input_width = 1440
        self.model = None

    def fit(self, X, y, epochs=None):
        X = self._reshape_data(X)
        input_shape, output_shape = self._data_shapes(X, y)

        if self.model is None:
            self._create_model(input_shape, output_shape)

        if epochs is None:
            epochs = self.epochs

        self.model.fit(X, y, batch_size=self.batch_size, epochs=epochs)
        cached_model_predict_clear()

        if self.dataloader:
            try:
                validation_data = self.dataloader.load_validate()
            except (NotImplementedError, AttributeError):
                validation_data = None

            if validation_data:
                self.validate(*validation_data)
            else:
                self.threshold(np.full(output_shape, .5))

    def validate(self, X, y):
        X = self._reshape_data(X)
        y_pred = self.model.predict(X)
        threshold = []
        for label_idx in range(y_pred.shape[1]):
            fpr, tpr, thresholds = roc_curve(y[..., label_idx],
                                             y_pred[..., label_idx])
            try:
                idx = find_elbow(tpr, fpr)
            except ValueError as ex:
                print(ex)
                idx = -1

            if idx >= 0:
                threshold.append(thresholds[idx])
            else:
                threshold.append(0.5)

        self.threshold = np.array(threshold)

    def _data_shapes(self, X, y):
        if X.shape[2] > self.network_input_width:
            raise ValueError('window_size > ' + str(self.network_input_width))
        input_shape = (X.shape[1], X.shape[2], X.shape[3])
        try:
            output_shape = y.shape[1]
        except IndexError:
            output_shape = 1

        return input_shape, output_shape

    def _create_model(self, input_shape, output_shape):
        melgram_input, output = self._cnn_layers(input_shape, output_shape)
        self.model = Model(inputs=melgram_input, outputs=output)
        self.model.compile(optimizer=RMSprop(lr=0.0001, decay=1e-6),
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
        self.model.summary()

    def _cnn_layers(self, input_shape, output_shape):
        channel_axis = 3

        melgram_input = Input(shape=input_shape, dtype='float32')

        # Input block
        padding = self.network_input_width - input_shape[1]
        left_pad = int(padding / 2)
        if padding % 2:
            right_pad = left_pad + 1
        else:
            right_pad = left_pad
        input_padding = ((0, 0), (left_pad, right_pad))
        hidden = ZeroPadding2D(padding=input_padding)(melgram_input)

        for idx, size in enumerate(self.block_sizes):
            hidden = conv_block(
                block_id=idx,
                filters=size,
                padding=self.padding,
                input_layer=hidden,
            )

        # reshaping
        hidden = Flatten()(hidden)
        hidden = Dense(
            256,
            name='dense1',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
        )(hidden)
        hidden = Activation('selu', name='selu6')(hidden)
        hidden = AlphaDropout(0.2)(hidden)
        hidden = Dense(
            output_shape,
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
        )(hidden)
        output = Activation('sigmoid')(hidden)

        return melgram_input, output

    def predict(self, X):
        predictions = self.predict_proba(X)
        labels = np.greater(predictions, self.threshold)

        return labels

    def predict_proba(self, X):
        X = self._reshape_data(X)
        return cached_model_predict(self.model, X)

    def _reshape_data(self, X):
        return X


def conv_block(block_id, filters, padding, input_layer):
    name = 'block-' + str(block_id) + '--'
    hidden = Conv2D(
        filters,
        (3, 3),
        padding=padding,
        name=name + 'conv-1',
        kernel_initializer='lecun_normal',
        bias_initializer='zeros',
    )(input_layer)
    hidden = Activation('selu', name=name + 'selu-1')(hidden)
    hidden = Conv2D(
        filters,
        (3, 3),
        name=name + 'conv-2',
        kernel_initializer='lecun_normal',
        bias_initializer='zeros',
    )(hidden)
    hidden = Activation('selu', name=name + 'selu-2')(hidden)
    hidden = MaxPooling2D(pool_size=(2, 2), name=name + 'pool-1')(hidden)
    output_layer = AlphaDropout(0.1, name=name + 'dropout-1')(hidden)

    return output_layer
