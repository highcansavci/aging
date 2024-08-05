from adfd_model.inference.inference import inference, check_face_availablilty
from PIL import Image
import io
import base64
import logging


class AgingService:

    @staticmethod
    def __numpy2base64(np_arr):
        # Convert NumPy array to PIL Image
        image = Image.fromarray(np_arr)
        # Save the PIL Image to a BytesIO object in PNG format
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        # Get the image data from the BytesIO object
        image_data = buffer.getvalue()
        # Encode the image data to Base64
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
        return base64_encoded

    @staticmethod
    def aging_task(np_arr):
        logging.info("Executing aging task in the adfd model service layer.")
        result = inference(np_arr)
        return {
            'age_10': AgingService.__numpy2base64(np_arr[0:1024, ...]),
            'age_30': AgingService.__numpy2base64(np_arr[1024:2048, ...]),
            'age_50': AgingService.__numpy2base64(np_arr[2048:3096, ...]),
            'age_70': AgingService.__numpy2base64(np_arr[3096:, ...])
        }

    @staticmethod
    def check_face_alignment(np_arr):
        logging.info(
            "Checking face alignment in the adfd model service layer.")
        return check_face_availablilty(np_arr)
