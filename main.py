from datetime import datetime

import requests
from pydantic import ValidationError
from source.Classes.RequestClass import Request


if __name__ == "__main__":
    try:
        req = Request(ticker="AMZN", starttime=datetime(2023, 2, 19), endtime=datetime(2023, 2, 28),
                      interval="1D", requesttime=datetime.now())
        analysis = req.analyzeRequest()
    except ValidationError as e:
        print(e)

