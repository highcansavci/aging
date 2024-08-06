import numpy as np
from PIL import Image
import base64
import torch
import io


def numpy2base64(np_arr):
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


def demo(image_path):
    return numpy2base64(np.array(Image.open(image_path)))


if __name__ == "__main__":
    with open("demo.txt", "+a", encoding="utf-8") as w:
        w.write(demo("face.png"))
