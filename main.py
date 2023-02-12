from datetime import datetime

from source.DataCollection import Request

if __name__ == "__main__":
    starttime = datetime(2021, 7, 25)
    endtime = datetime(2021, 8, 25)
    request = Request(
        ticker="MSFT",
        starttime=starttime,
        endtime=endtime,
        interval="1H",
    )
    print(request)
