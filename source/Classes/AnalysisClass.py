from datetime import datetime
from pydantic import BaseModel, ValidationError, root_validator, validator

from source.Classes.RequestClass import Request


class Analysis(Request):
    ticker: str
    starttime: datetime
    endtime: datetime
    interval: str
    errorcodes: list

    def __init__(self, request, **data: Any):
        super().__init__(**data)
        self.starttime = request.starttime
        self.endtime = request.endtime
        self.ticker = request.ticker
        self.interval = request.interval

    @validator("endtime")
    def checkDates(cls, value, values):
        if value < values["starttime"]:
            values["errorcodes"].append("timeframeError")
            raise ValueError("endtime must be greater than or equal to starttime")
        return value

