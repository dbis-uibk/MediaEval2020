import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_curve
import tensorflow.compat.v2.keras.backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GRU
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


class CRNNModel(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 batch_size=64,
                 epochs=100,
                 padding='same',
                 dataloader=None,
                 output_dropout=0.3,
                 attention=False):
        self.batch_size = batch_size
        self.epochs = epochs
        self.padding = padding
        self.dataloader = dataloader
        self.output_dropout = output_dropout
        self.attention = attention
        self.label_split = None
        self.network_input_width = 1440
        self.model = None
        self.threshold = None

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

            if validation_data and self.label_split is not None:
                data, labels = validation_data
                assert len(self.label_split) == output_shape
                self.validate(data, labels[..., self.label_split])
            elif validation_data and self.label_split is None:
                labels = validation_data[1]
                try:
                    assert labels.shape[1] == output_shape
                except IndexError:
                    assert output_shape == 1
                self.validate(*validation_data)
            else:
                self.threshold = np.full(output_shape, .5)

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
        output_shape = y.shape[1]

        return input_shape, output_shape

    def _create_model(self, input_shape, output_shape):
        melgram_input, output = self._crnn_layers(input_shape, output_shape)
        self.model = Model(inputs=melgram_input, outputs=output)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
        self.model.summary()

    def _crnn_layers(self, input_shape, output_shape):
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

        # Conv block 1
        hidden = Conv2D(64, (3, 3), padding=self.padding, name='conv1')(hidden)
        hidden = BatchNormalization(axis=channel_axis, name='bn1')(hidden)
        hidden = Activation('elu', name='elu-1')(hidden)
        hidden = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                              name='pool1')(hidden)
        hidden = Dropout(0.1, name='dropout1')(hidden)

        # Conv block 2
        hidden = Conv2D(128, (3, 3), padding=self.padding,
                        name='conv2')(hidden)
        hidden = BatchNormalization(axis=channel_axis, name='bn2')(hidden)
        hidden = Activation('elu', name='elu-2')(hidden)
        hidden = MaxPooling2D(pool_size=(3, 3), strides=(3, 3),
                              name='pool2')(hidden)
        hidden = Dropout(0.1, name='dropout2')(hidden)

        # Conv block 3
        hidden = Conv2D(128, (3, 3), padding=self.padding,
                        name='conv3')(hidden)
        hidden = BatchNormalization(axis=channel_axis, name='bn3')(hidden)
        hidden = Activation('elu', name='elu-3')(hidden)
        hidden = MaxPooling2D(pool_size=(4, 4), strides=(4, 4),
                              name='pool3')(hidden)
        hidden = Dropout(0.1, name='dropout3')(hidden)

        # Conv block 4
        hidden = Conv2D(128, (3, 3), padding=self.padding,
                        name='conv4')(hidden)
        hidden = BatchNormalization(axis=channel_axis, name='bn4')(hidden)
        hidden = Activation('elu', name='elu-4')(hidden)
        hidden = MaxPooling2D(pool_size=(4, 4), strides=(4, 4),
                              name='pool4')(hidden)
        hidden = Dropout(0.1, name='dropout4')(hidden)

        # reshaping
        hidden = Reshape((15, 128))(hidden)

        # GRU block 1, 2, output
        embed_size = 32
        hidden = GRU(embed_size, return_sequences=True, name='gru1')(hidden)
        hidden = GRU(embed_size, return_sequences=self.attention,
                     name='gru2')(hidden)

        if self.attention:
            attention = Dense(1)(hidden)
            attention = Flatten()(attention)
            attention_act = Activation('softmax')(attention)
            attention = RepeatVector(embed_size)(attention_act)
            attention = Permute((2, 1))(attention)

            merged = Multiply()([hidden, attention])
            hidden = Lambda(lambda xin: K.sum(xin, axis=1))(merged)

        if self.output_dropout:
            hidden = Dropout(self.output_dropout)(hidden)
        output = Dense(output_shape, activation='sigmoid',
                       name='crnn_output')(hidden)

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


class CRNNPlusModel(CRNNModel):

    def __init__(self,
                 batch_size=64,
                 epochs=100,
                 padding='same',
                 dataloader=None,
                 output_dropout=0.3,
                 concat_bn=False,
                 attention=False):
        super().__init__(batch_size=batch_size,
                         epochs=epochs,
                         padding=padding,
                         dataloader=dataloader,
                         output_dropout=output_dropout,
                         attention=attention)
        self.concat_bn = concat_bn

    def _data_shapes(self, X, y):
        mel_shape = X[0][0].shape
        ess_shape = X[1][0].shape

        if mel_shape[1] > self.network_input_width:
            raise ValueError('window_size > ' + str(self.network_input_width))

        input_shape = (mel_shape, ess_shape)
        output_shape = y.shape[1]

        return input_shape, output_shape

    def _reshape_data(self, X):
        return list(zip(*X))

    def _create_model(self, input_shape, output_shape):
        mel_input, crnn_output = self._crnn_layers(input_shape[0],
                                                   output_shape)
        essentia_input = Input(shape=input_shape[1], dtype='float32')

        concat = Concatenate()([crnn_output, essentia_input])
        if self.concat_bn:
            concat = BatchNormalization(axis=-1, name='concat_bn')(concat)

        # Dense
        dense = Dense(128, activation='tanh', name='dense_10')(concat)
        output = Dense(output_shape, activation='sigmoid',
                       name='output')(dense)

        self.model = Model(inputs=[mel_input, essentia_input], outputs=output)
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()
