import sys
import os

# Determine the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the project root to sys.path
sys.path.append(project_root)

from backend.log.config import setup_logging
from backend.adfd.router.aging_router import router as adfd_aging_router
from backend.sam.router.aging_router import router as sam_aging_router
from fastapi import FastAPI


app = FastAPI()
setup_logging()
app.include_router(sam_aging_router)
app.include_router(adfd_aging_router)
