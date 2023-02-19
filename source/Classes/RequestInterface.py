import json
import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import requests
from pandas import DataFrame
from pydantic import BaseModel, validator
import matplotlib.pyplot as plt

from source.DataCollection.RequestDataCollection import dataCollector
from source.NeuralNetwork.LSTMConfig.ModelService import getConfigAndData, modelInit, prediction, getTrainXY

key = 'JX8SQV1M7PTAB6YB'


@dataclass
class AnalysisResult:
    image: str
    pred: DataFrame | int
    errors: str


class Request(BaseModel):
    ticker: str
    starttime: datetime
    endtime: datetime
    requesttime: datetime
    interval: str

    @validator("endtime")
    def checkDates(cls, value, values):
        if value < values["starttime"]:
            raise ValueError("endtime must be greater than or equal to starttime")
        return value

    @validator("interval")
    def checkInterval(cls, value):
        intervalList = ['1D', '3D', '1W', '1M']
        if value not in intervalList:
            raise ValueError('Wrong interval')
        return value

    @validator("ticker")
    def checkTicker(cls, value):
        tickers = []
        urlTickerCheck = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={value}&apikey={key}'
        data = requests.get(urlTickerCheck).json()
        for el in data['bestMatches']:
            tickers.append(el['1. symbol'])
        if value not in tickers:
            raise ValueError('Ticker does not exist')
        else:
            return value

    def analyze_request(self) -> dict:
        analysisData = self.analysis_process()
        analysis = {'image': analysisData.image, 'pred': analysisData.pred,
                    'errors': analysisData.errors}
        return analysis

    def analysis_process(self) -> AnalysisResult:
        pred = generate_pred(self)

        if isinstance(pred, pd.DataFrame):
            image, error = generate_image(self, pred)
            return AnalysisResult(image=image, pred=pred, errors=error)
        elif isinstance(pred, Exception):
            return AnalysisResult(image=f'source/Classes/Images/error.jpg', pred=0, errors='data generation error')


def generate_image(RequestObject, pred) -> [str, str]:
    ticker = RequestObject.ticker
    starttime = RequestObject.starttime.strftime('%Y-%m-%d')
    endtime = RequestObject.endtime.strftime('%Y-%m-%d')
    data = pred
    image = f'source/Classes/Images/request_{ticker}_{starttime}_{endtime}.jpg'
    lst = check_for_image(data)
    isok = lst[0]
    error = lst[1]
    if isok:
        plt.plot(data, label=RequestObject.ticker)
        plt.savefig(image)
        plt.show()
    else:
        image = f'source/Classes/Images/error.jpg'

    return [image, error]


def generate_pred(RequestObject) -> dict[datetime, float] | Exception:
    data = dataCollector(RequestObject)
    configs, dataConfig = getConfigAndData(data)
    model = modelInit(configs)
    x, y = getTrainXY(dataConfig, configs)
    prediction(configs, model, dataConfig, x, y)

    return data


def check_for_image(data):
    if len(data) > 5:
        ret = [True, 'ok']
    else:
        ret = [False, 'data error']
    return ret
