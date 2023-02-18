from datetime import datetime

from source.Classes.RequestInterface import Request

if __name__ == "__main__":
    req = Request("MSFT", datetime(2023, 1, 17), datetime(2023, 2, 17), "1D")
    analysis = req.analyze_request()
