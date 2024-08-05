from fastapi import APIRouter
import base64
import io
from PIL import Image
import numpy as np
from sam.controller.aging_controller import AgingController
import logging

router = APIRouter()


@router.post("/api/aging/sam_model")
def execute_aging(base64_img):
    logging.info(
        "Converting base 64 string to numpy array in the sam model router layer.")
    # Convert base 64 image to numpy array
    # Decode the Base64 string
    image_data = base64.b64decode(base64_img)

    # Convert binary data to a BytesIO object
    image_bytes = io.BytesIO(image_data)

    # Open the image using Pillow
    image = Image.open(image_bytes)

    # Convert the image to a NumPy array
    numpy_array = np.array(image)
    logging.info(
        "Checking if any face is detected in the sam model router layer.")
    if not AgingController.check_face_alignment(numpy_array):
        return {'error': 'The face is not found.'}
    logging.info(
        "Starting to execute aging task in the sam model router layer.")
    return AgingController.aging_task(numpy_array)
