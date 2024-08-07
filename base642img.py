from PIL import Image
import base64
import io
import json


def base642img(base64_img, age):
    image_data = base64.b64decode(base64_img)

    # Convert binary data to a BytesIO object
    image_bytes = io.BytesIO(image_data)

    # Open the image using Pillow
    image = Image.open(image_bytes).convert("RGB")
    image.save(f"result_{str(age)}.png")


if __name__ == "__main__":
    TARGET_AGES = [10, 30, 50, 70]
    with open('results.json', 'r') as file:
        data = json.load(file)
        for age in TARGET_AGES:
            base642img(data[f"age_{age}"], age)
