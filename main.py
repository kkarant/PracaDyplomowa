from datetime import datetime

import requests
from pydantic import ValidationError
from source.Classes.RequestInterface import Request


if __name__ == "__main__":
    try:
        req = Request(ticker="AMZN", starttime=datetime(2022, 5, 17), endtime=datetime(2023, 2, 17),
                      interval="1D", requesttime=datetime.now())
        analysis = req.analyze_request()
    except ValidationError as e:
        print(e)

