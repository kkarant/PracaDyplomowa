import json
import math
import os
from datetime import datetime
from matplotlib import pyplot as plt
from source.DataCollection.RequestDataCollection import DataObject, dataCollector
from source.NeuralNetwork.LSTMConfig.Model import Model


def getConfig():
    return json.load(open('source/NeuralNetwork/LSTMConfig/LSTMconfig.json', 'r'))


def generate_pred(RequestObject) -> dict[datetime, float] | Exception:
    config = getConfig()
    dataObject = getDataObject(dataCollector(RequestObject)['normalized'],
                               dataCollector(RequestObject)['source'], config)

    model = modelInit(config)
    model_name = trainModel(config, model, dataObject)  # needs full remake
    model_results = testModel(model, dataObject)  # needs to be written

    predictions = predict(model_name, dataObject)
    return predictions


def plot_results(predicted_data):  # , true_data
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    # ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction', color='red')
    plt.legend()
    plt.show()


def getDataObject(data_normalized, data_source, configs) -> DataObject:
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    dataObject = DataObject(
        data_normalized,
        data_source,
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    return dataObject


def modelInit(configs) -> Model:
    model = Model()
    model.build_model(configs)
    return model


# TODO remake because train generator govno
def trainModel(configs, model, data) -> str:
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model_name = model.train_generator(
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

    return model_name


# TODO write test to return rmse mse and pred for image generation (for dev)
def testModel(model, dataObject):
    ...


# TODO write prdict for n points into the future, return type not sure
def predict(model_name: str, dataObject: DataObject) -> dict[datetime, float]:
    ...
