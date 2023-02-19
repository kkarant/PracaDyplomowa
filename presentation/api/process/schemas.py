from datetime import datetime

from pydantic import BaseModel


class Request(BaseModel):
    ticker: str
    starttime: datetime
    endtime: datetime
    requesttime: datetime
    interval: str
