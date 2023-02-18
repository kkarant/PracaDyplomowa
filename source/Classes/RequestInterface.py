from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, ValidationError, root_validator, validator
import matplotlib.pyplot as plt
import yfinance as yf


@dataclass
class AnalysisResult:
    image: str
    pred: dict
    errors: list[str]


class Request:
    def __init__(self, ticker: str, starttime: datetime, endtime: datetime, interval: str):
        self.ticker = ticker
        self.starttime = starttime
        self.endtime = endtime
        self.requesttime = datetime.now()
        self.interval = interval
        self.errors = []

    @validator("endtime")
    def checkDates(cls, value, values):
        if value < values["starttime"]:
            values['errors'].append('dateError')
            raise ValueError("endtime must be greater than or equal to starttime")
        return value

    @validator("interval")
    def checkInterval(cls, value, values):
        intervalList = ['1D', '3D', '1W', '1M']
        if value not in intervalList:
            values['errors'].append('intervalError')
            raise ValueError('Wrong interval')
        return value

    @validator("ticker")
    def checkTicker(cls, value, values):
        y = yf.Ticker(value)
        if y == 1:
            return value
        elif y == 0:
            values['errors'].append('tickerError')
            raise ValueError('Ticker does not exist')

    def analyze_request(self) -> dict:
        analysisData = self.analysis_process()
        analysis = {'image': analysisData.image, 'dictionary': analysisData.pred,
                    'errors': analysisData.errors}
        return analysis

    def analysis_process(self) -> AnalysisResult:
        pred = generate_dictionary(self)
        image = generate_image(self, pred)
        errors = self.errors
        return AnalysisResult(image=image, pred=pred, errors=errors)


def generate_image(RequestObject, pred) -> str:
    ticker = RequestObject.ticker
    starttime = RequestObject.starttime.strftime('%Y-%m-%d')
    endtime = RequestObject.endtime.strftime('%Y-%m-%d')
    data = pred
    image = f'source/Classes/Images/request_{ticker}_{starttime}_{endtime}.jpg'
    plt.plot(data, label=RequestObject.ticker)
    plt.savefig(image)
    plt.show()

    return image


def generate_dictionary(RequestObject) -> dict[datetime, float]:
    data = yf.download(tickers=RequestObject.ticker, interval=RequestObject.interval,
                       start=RequestObject.starttime, end=RequestObject.endtime)
    dictionary = data[['Close']]
    return dictionary


def generate_errors(RequestObject) -> list[str]:
    return RequestObject.errors
