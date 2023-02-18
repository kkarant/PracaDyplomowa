from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import requests
from pydantic import BaseModel, ValidationError, root_validator, validator
import matplotlib.pyplot as plt
import yfinance as yf

key = 'JX8SQV1M7PTAB6YB'


@dataclass
class AnalysisResult:
    image: str
    pred: dict
    errors: list[str]


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
    def checkInterval(cls, value, values):
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
        print(tickers)
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
        errors = []
        pred = generate_dictionary(self)

        if isinstance(pred, pd.DataFrame):
            image, errors_image = generate_image(self, pred)
        elif isinstance(pred, Exception):
            errors = 'data generation error'
            image = f'source/Classes/Images/error.jpg'

        return AnalysisResult(image=image, pred=pred, errors=errors)


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


def generate_dictionary(RequestObject) -> dict[datetime, float] | Exception:
    try:
        data = yf.download(tickers=RequestObject.ticker, interval=RequestObject.interval,
                           start=RequestObject.starttime, end=RequestObject.endtime)
        dictionary = data[['Close']]
        return dictionary
    except Exception as e:
        return e


def check_for_image(data):
    if len(data) > 5:
        ret = [True, 'ok']
    else:
        ret = [False, 'data error']
    return ret
