import os
import numpy as np
from datetime import datetime
from numpy import newaxis
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Model:
    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs, neurons_output):
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons_output, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')

    def train_model(self, configs, data, validation_split=0.1):
        epochs = configs['training']['epochs']
        print(epochs)
        batch_size = configs['training']['batch_size']
        save_dir = configs['model']['save_dir']
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        callbacks = [
            EarlyStopping(monitor='loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]

        history = self.model.fit(x=data.data_train, y=data.source_data_train, epochs=epochs,
                                 batch_size=batch_size, callbacks=callbacks, validation_split=validation_split)

        self.model.save(save_fname)
        print(history)
        # opt = SGD(lr=0.01, momentum=0.9)
        # model.compile(loss='mean_squared_error', optimizer=opt)

        return history, save_dir

