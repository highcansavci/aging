from fastapi import FastAPI
from sam.router.aging_router import router as sam_aging_router
from adfd.router.aging_router import router as adfd_aging_router
from log.config import setup_logging

app = FastAPI()
setup_logging()
app.include_router(sam_aging_router)
app.include_router(adfd_aging_router)

