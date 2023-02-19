from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def normalise_windows(window_data, single_window=False):
    normalised_data = []
    window_data = [window_data] if single_window else window_data
    for window in window_data:
        normalised_window = []
        for col_i in range(window.shape[1]):
            normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
            normalised_window.append(normalised_col)
        normalised_window = np.array(normalised_window).T
        normalised_data.append(normalised_window)
    return np.array(normalised_data)


class DataPrep:
    def __init__(self, data, split, cols):
        i_split = int(len(data) * split)
        self.data_train = data.get(cols).values[:i_split]
        self.data_test = data.get(cols).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        data_windows = []
        print(self.len_test)
        print(seq_len)
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])
        data_windows = np.array(data_windows).astype(float)
        data_windows = normalise_windows(data_windows, single_window=False) if normalise else data_windows
        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y

    def get_train_data(self, seq_len, normalise):
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        window = self.data_train[i:i+seq_len]
        window = normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y


def dataCollector(RequestObject) -> pd.DataFrame | Exception:
    tickers = RequestObject.ticker
    interval = RequestObject.interval
    start = RequestObject.starttime
    end = RequestObject.endtime
    trainStart = start - relativedelta(years=2)
    trainEnd = datetime.now()
    try:
        data = yf.download(tickers=tickers, interval=interval,
                           start=trainStart, end=trainEnd)
        dictionary = data
        return dictionary
    except Exception as e:
        return e


def dataNormalization(data: pd.DataFrame) -> pd.Series:
    ser = pd.Series(data.reset_index()['Close'], copy=False)
    values = ser.values.reshape(-1, 1)
    print(type(ser))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

    normalized = scaler.transform(values)
    # print(scaler.inverse_transform(normalized))
    return normalized


