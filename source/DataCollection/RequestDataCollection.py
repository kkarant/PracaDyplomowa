from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataObject:
    def __init__(self, data_normalized, data_source, split, cols):
        split = int(len(data_normalized) * split)
        self.data_train = data_normalized.get(cols).values[:split]
        self.data_test = data_normalized.get(cols).values[split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.source_data_test = data_source.get(cols).values[split:]
        self.source_data_len_test = len(self.data_test)


def dataCollector(RequestObject) -> dict[str, pd.DataFrame] | Exception:  # TODO return types after normalization
    tickers = RequestObject.ticker
    interval = RequestObject.interval
    start = RequestObject.starttime
    end = RequestObject.endtime
    trainStart = start - relativedelta(years=3)
    trainEnd = datetime.now()
    try:
        data = yf.download(tickers=tickers, interval=interval,
                           start=trainStart, end=trainEnd)
        dictionary = data
        dictionary = dataNormalization(dictionary)
        return {'source': data, 'normalized': dictionary}
    except Exception as e:
        return e


def dataNormalization(data: pd.DataFrame):
    ser = pd.Series(data.reset_index(), copy=False)
    values = ser.values.reshape(-1, 1)
    print(type(ser))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    # print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

    normalized = scaler.transform(values)
    # print(scaler.inverse_transform(normalized))
    return normalized
