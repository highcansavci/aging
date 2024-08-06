from PIL import Image
import argparse
import base64
import io


def base642img(base64_img):
    image_data = base64.b64decode(base64_img)

    # Convert binary data to a BytesIO object
    image_bytes = io.BytesIO(image_data)

    # Open the image using Pillow
    image = Image.open(image_bytes)
    image.save("result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a Base64 encoded image.")
    parser.add_argument("base64_img", type=str,
                        help="Base64 encoded image string")

    args = parser.parse_args()
    base642img(args.base64_img)
