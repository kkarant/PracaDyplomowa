import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .process import controllers


def create_app() -> FastAPI:
    app = FastAPI()
    include_routers(app)
    include_static(app)

    return app


def include_routers(app: FastAPI) -> None:
    app.include_router(controllers.router)


def include_static(app: FastAPI) -> None:
    os.makedirs('./data/', exist_ok=True)

    app.mount('/data/', StaticFiles(directory='data'), name='data')
