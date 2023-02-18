from datetime import datetime

from source.Classes.RequestClass import Request
from source.Classes.RequestInterface import RequestService

if __name__ == "__main__":
    DTOObject = ["MSFT", datetime(2021, 7, 25), datetime(2021, 8, 25), "1H"]

    RequestService.analysisRequest(DTOObject)
