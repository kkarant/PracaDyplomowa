import json
import os
import pandas as pd

from datetime import datetime
from matplotlib import pyplot as plt
from source.DataCollection.RequestDataCollection import DataObject, dataCollector
from source.NeuralNetwork.LSTMConfig.Model import Model


def getConfig():
    return json.load(open('source/NeuralNetwork/LSTMConfig/LSTMconfig.json', 'r'))


def generate_pred(RequestObject) -> pd.DataFrame | Exception:
    config = getConfig()
    dict = dataCollector(RequestObject)
    dataObject = getDataObject(dict['normalized'],
                               dict['source'], config)
    print(f'train len: {len(dataObject.data_train)}, test len: {len(dataObject.data_test)}')
    model = modelInit(config, RequestObject.getNumberOfSteps())
    history, save_dir = model.train_model(config, dataObject, validation_split=0.075)  # needs full remake
    print(history)
    # model_results = testModel(model, dataObject)  # needs to be written
    #
    # predictions = predict(model_name, dataObject)
    return pd.DataFrame()


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
    )
    return dataObject


def modelInit(configs, neurons_output) -> Model:
    model = Model()
    model.build_model(configs, neurons_output)
    return model


# TODO write test to return rmse mse and pred for image generation (for dev)
def testModel(model, dataObject):
    ...


# TODO write prdict for n points into the future, return type not sure
def predict(model_name: str, dataObject: DataObject) -> dict[datetime, float]:
    ...
