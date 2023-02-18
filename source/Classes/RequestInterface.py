from datetime import datetime

from source.Classes.AnalysisClass import Analysis
from source.Classes.RequestClass import Request




class RequestService:
    @staticmethod
    def analysisRequest(DTOObject) -> Analysis:
        request = Request(DTOObject)
        analysisObject = Analysis(request)
        return analysisObject
