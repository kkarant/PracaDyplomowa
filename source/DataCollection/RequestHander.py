import pandas as pd
import yfinance as yf


class RequestService(Protocol):
    @staticmethod
    def analysisRequest(DTOObject):
        request = Request(DTOObject)
        analysisObject = Analysis(request)
        return analysisObject

