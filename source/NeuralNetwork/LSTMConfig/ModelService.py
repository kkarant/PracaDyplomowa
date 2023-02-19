import json
import math
import os

import numpy as np
from matplotlib import pyplot as plt

from source.DataCollection.RequestDataCollection import DataPrep
from source.NeuralNetwork.LSTMConfig.Model import Model


def plot_results(predicted_data): # , true_data
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    # ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction', color='red')
    plt.legend()
    plt.show()


def getConfigAndData(data):
    configs = json.load(open('source/NeuralNetwork/LSTMConfig/LSTMconfig.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    dataConfig = DataPrep(
        data,
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    return configs, dataConfig


def modelInit(configs):
    model = Model()
    model.build_model(configs)
    return model


def getTrainXY(data, configs):
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    return x, y


def prediction(configs, model, data, x, y, RequestObject):
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )
    X = data.data_train
    n_steps = RequestObject.getNumberOfSteps()
    inputs = X[-1].reshape(1, -1)
    preds = []
    for i in range(n_steps):
        pred = model.predict(inputs)[0]
        preds.append(pred)
        inputs = np.concatenate([inputs[:, 1:], pred.reshape(1, -1)], axis=1)
    # x_test, y_test = data.get_pred_window(
    #     seq_len=RequestObject.starttime,
    #     normalise=RequestObject.endtime
    # )
    #
    # # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
    # #                                                configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # # predictions = model.predict_point_by_point(x_test)

    plot_results(preds)
