from backend.log.config import setup_logging
from backend.adfd.router.aging_router import router as adfd_aging_router
from backend.sam.router.aging_router import router as sam_aging_router
from fastapi import FastAPI
import os
import sys

# Set the working directory to the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Append the project root to sys.path
sys.path.append(os.getcwd())


app = FastAPI()
setup_logging()
app.include_router(sam_aging_router)
app.include_router(adfd_aging_router)
