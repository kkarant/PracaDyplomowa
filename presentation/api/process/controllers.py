from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from .schemas import Request as ProcessingRequest

from source.Classes.RequestInterface import Request as RequestService

router = APIRouter(prefix="/process", tags=["processing"])


@router.post('/')
def process_data(data_to_process: ProcessingRequest) -> dict:
    """
    Some Documentation
    """

    try:
        process_result = RequestService(
            ticker=data_to_process.ticker,
            starttime=data_to_process.starttime,
            endtime=data_to_process.endtime,
            requesttime=data_to_process.requesttime,
            interval=data_to_process.interval
        ).analyze_request()
    except ValidationError as err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(err))

    return process_result
