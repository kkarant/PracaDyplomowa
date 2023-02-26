from datetime import datetime

import requests
from pydantic import ValidationError
from source.Classes.RequestClass import Request


if __name__ == "__main__":
    try:
        req = Request(ticker="AMZN", starttime=datetime(2023, 2, 26), endtime=datetime(2023, 3, 7),
                      interval="1D", requesttime=datetime.now())
        analysis = req.analyzeRequest()
    except ValidationError as e:
        print(e)

