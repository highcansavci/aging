from fastapi import FastAPI
from backend.sam.router.aging_router import router as aging_router
from backend.log.config import setup_logging

app = FastAPI()
setup_logging()
app.include_router(aging_router)
