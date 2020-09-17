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
                 epochs=100,
                 padding='same',
                 dataloader=None,
                 output_dropout=0.3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.padding = padding
        self.dataloader = dataloader
        self.network_input_width = 1440

    def fit(self, X, y):
        X = self._reshape_data(X)
        input_shape, output_shape = self._data_shapes(X, y)
        self._create_model(input_shape, output_shape)

        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)
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
                           loss="categorical_crossentropy",
                           metrics=['accuracy'])
        self.model.summary()

    def _cnn_layers(self, input_shape, output_shape):
        channel_axis = 3

        melgram_input = Input(shape=input_shape, dtype="float32")

        # Input block
        padding = self.network_input_width - input_shape[1]
        left_pad = int(padding / 2)
        if padding % 2:
            right_pad = left_pad + 1
        else:
            right_pad = left_pad
        input_padding = ((0, 0), (left_pad, right_pad))
        hidden = ZeroPadding2D(padding=input_padding)(melgram_input)

        # Conv block 1
        hidden = Conv2D(
            32,
            (3, 3),
            padding=self.padding,
            name='conv1-1',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
        )(hidden)
        hidden = Activation('selu', name='selu1-1')(hidden)
        hidden = Conv2D(
            32,
            (3, 3),
            name='conv1-2',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
        )(hidden)
        hidden = Activation('selu', name='selu1-2')(hidden)
        hidden = MaxPooling2D(pool_size=(2, 2), name='pool1')(hidden)
        hidden = AlphaDropout(0.1, name='dropout1')(hidden)

        # Conv block 2
        hidden = Conv2D(
            64,
            (3, 3),
            name='conv2-1',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
        )(hidden)
        hidden = Activation('selu', name='selu2-1')(hidden)
        hidden = Conv2D(
            64,
            (3, 3),
            name='conv2-2',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
        )(hidden)
        hidden = Activation('selu', name='selu2-2')(hidden)
        hidden = MaxPooling2D(pool_size=(2, 2), name='pool2')(hidden)
        hidden = AlphaDropout(0.1, name='dropout2')(hidden)

        # Conv block 3
        hidden = Conv2D(
            64,
            (3, 3),
            name='conv3-1',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
        )(hidden)
        hidden = Activation('selu', name='selu3-1')(hidden)
        hidden = Conv2D(
            64,
            (3, 3),
            name='conv3-2',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
        )(hidden)
        hidden = Activation('selu', name='selu3-2')(hidden)
        hidden = MaxPooling2D(pool_size=(2, 2), name='pool3')(hidden)
        hidden = AlphaDropout(0.1, name='dropout3')(hidden)

        # Conv block 4
        hidden = Conv2D(
            64,
            (3, 3),
            name='conv4-1',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
        )(hidden)
        hidden = Activation('selu', name='selu4-1')(hidden)
        hidden = Conv2D(
            64,
            (3, 3),
            name='conv4-2',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
        )(hidden)
        hidden = Activation('selu', name='selu4-2')(hidden)
        hidden = MaxPooling2D(pool_size=(2, 2), name='pool4')(hidden)
        hidden = AlphaDropout(0.1, name='dropout4')(hidden)

        # Conv block 5
        hidden = Conv2D(
            64,
            (3, 3),
            name='conv5-1',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
        )(hidden)
        hidden = Activation('selu', name='selu5-1')(hidden)
        hidden = Conv2D(
            64,
            (3, 3),
            name='conv5-2',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
        )(hidden)
        hidden = Activation('selu', name='selu5-2')(hidden)
        hidden = MaxPooling2D(pool_size=(2, 2), name='pool5')(hidden)
        hidden = AlphaDropout(0.1, name='dropout5')(hidden)
        # reshaping
        hidden = Flatten()(hidden)
        hidden = Dense(
            512,
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
        output = Activation('softmax')(hidden)

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
