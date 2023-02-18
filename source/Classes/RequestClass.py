from datetime import datetime

from pydantic import BaseModel, ValidationError, root_validator, validator


class Request(BaseModel):
    ticker: str
    starttime: datetime
    endtime: datetime
    interval: str

    @validator("endtime")
    def checkDates(cls, value, values):
        if value < values["starttime"]:
            raise ValueError("endtime must be greater than or equal to starttime")
        return value