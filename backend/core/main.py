from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi_injector import attach_injector

from api import router

from .di import create_injector
from .log_config import setup_logging

load_dotenv()
setup_logging()
app = FastAPI(title="CinemAI Recommendation API")

injector = create_injector()
attach_injector(app, injector)
app.include_router(router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
