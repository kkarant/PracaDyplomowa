from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataObject:
    def __init__(self, data_normalized, data_source, split):
        split = int(len(data_normalized) * split)
        self.data_train = data_normalized[:split]
        self.data_test = data_normalized[split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.source_data_train = data_source[:split]
        self.source_data_test = data_source[split:]
        self.source_data_len_test = len(self.data_test)


def dataCollector(RequestObject) -> dict[str, np.ndarray] | Exception:
    ticker = RequestObject.ticker
    interval = RequestObject.interval
    start = RequestObject.starttime
    end = RequestObject.endtime
    trainStart = start - relativedelta(years=3)
    trainEnd = datetime.now()
    try:
        data = yf.download(tickers=ticker, interval=interval,
                           start=trainStart, end=trainEnd)
        index_list, normalized, data = dataNormalization(data)
        return {'source': data, 'normalized': normalized}
    except Exception as e:
        return e


def dataNormalization(data: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    index_list = list(data.index.values)
    data = data.reset_index(drop=True)
    data = data['Close']
    normalized = scaler.fit_transform(data.values.reshape(-1,1))
    # print(scaler.inverse_transform(normalized))
    return index_list, normalized, data.to_numpy()
