import traceback
import logging
from backend.sam.controller.aging_controller import AgingController
import numpy as np
from PIL import Image
import io
import base64
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Request

router = APIRouter()


@router.post("/api/aging/sam_model")
async def execute_aging(request: Request):
    data = await request.json()
    base64_img = data.get('base64_img')

    if not base64_img:
        return JSONResponse(
            content={"detail": [{"type": "missing", "loc": [
                "body", "base64_img"], "msg": "Field required", "input": None}]},
            status_code=400
        )

    try:
        logging.info(
            "Converting base 64 string to numpy array in the sam model router layer.")
        # Convert base 64 image to numpy array
        # Decode the Base64 string
        image_data = base64.b64decode(base64_img)

        # Convert binary data to a BytesIO object
        image_bytes = io.BytesIO(image_data)

        # Open the image using Pillow
        image = Image.open(image_bytes).convert("RGB")

        # Convert the image to a NumPy array
        numpy_array = np.array(image)
        logging.info(
            "Checking if any face is detected in the sam model router layer.")
        if not AgingController.check_face_alignment(numpy_array):
            return JSONResponse(content={'error': 'The face is not found.'}, status_code=400)
        logging.info(
            "Starting to execute aging task in the sam model router layer.")
        return JSONResponse(content=AgingController.aging_task(numpy_array), status_code=200)
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        traceback.print_exc()
        return JSONResponse(
            content={"detail": [{"type": "error", "msg": str(e)}]},
            status_code=500
        )
