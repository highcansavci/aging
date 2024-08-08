from backend.log.config import setup_logging
from backend.adfd.router.aging_router import router as adfd_aging_router
from backend.sam.router.aging_router import router as sam_aging_router
from fastapi import FastAPI


app = FastAPI()
setup_logging()
app.include_router(sam_aging_router)
app.include_router(adfd_aging_router)
