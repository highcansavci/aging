# Aging Transformation API

This project provides an API for face aging transformation using two models: ADFD and SAM. The backend is developed in Python, utilizing FastAPI, and deployed on a Google Cloud Platform (GCP) instance with Docker.
It is seen that SAM has the better speed and accuracy compared to ADFD.

## Project Definition

This service generates four distinct aged versions (like 10, 30, 50, 70) of a given input image and will be accessible through FastAPI. It is written in Python, dockerized and deployed to Google Cloud Platform.

## FastAPI Endpoint Requirements

The FastAPI endpoint is working as explained below:

1. **Detected Face in Input Image**: Validating that the input image contains a detectable face. If no face is found, it will return an error. If a face is detected, crop the face.
2. **Handle Base64 Image Input**: Accepting an input image in Base64 format.
3. **Output Four Aged Versions**: Returning the aged images in Base64 format, each representing a different age stage.
4. **Dockerize**: Packaging the application in a Docker container for easy deployment and scalability.
5. **Use Open Source Models**: Leveraging open-source models for face detection and aging transformation.
6. **Logging System**: Implementing a logging system to track application events and errors.

## Input and Output Specifications

**Input:**

- **Base64 Image**: The input image provided by the user that needs to be aged.

**Output:**

- **Base64 Images**: Four distinct aged versions of the input image, each depicting the face at ages 10, 30, 50, and 70.

## Implementation Steps

1. **Dockerization**:
    - Created a Dockerfile for the FastAPI application.
    - Ensured all dependencies, including the necessary models for face detection and aging transformations, are installed within the Docker container.
    - Configured Docker Compose if needed for multi-container setups.

2. **Used Open Source Models**:
    - Integrated open-source face detection models (e.g. Dlib and InsightFace).
    - Utilized open-source aging transformation models (e.g., SAM and ADFD). Compared the models in terms of accuracy and speed.

## Prerequisites
* Linux or macOS
* NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
* Python 3.12

## Repository

The project repository is hosted on GitHub: [https://github.com/highcansavci/aging](https://github.com/highcansavci/aging)

## Endpoints

- **SAM Model**: [http://34.91.249.118:80/api/aging/sam_model](http://34.91.249.118:80/api/aging/sam_model)
- **ADFD Model**: [http://34.91.249.118:80/api/aging/adfd_model](http://34.91.249.118:80/api/aging/adfd_model)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/highcansavci/aging.git
    cd aging
    ```

2. Build the Docker image:

    ```bash
    docker build -t aging .
    ```

3. Run the Docker container:

    ```bash
    docker run --gpus all -p 80:8000 -v /usr/local/cuda-12.1:/usr/local/<cuda> aging
    ```

## Docker Compose

You can also use Docker Compose to manage the container. Created a `docker-compose.yaml` file with the following content:

```yaml
version: "3.8"

services:
  aging_service:
    image: karabairak/aging:latest
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "80:8000"
    volumes:
      - /usr/local/cuda-12.1:/usr/local/<cuda>
    environment:
      CUDA_HOME: /usr/local/<cuda>
      PYTHONDONTWRITEBYTECODE: 1
      PYTHONUNBUFFERED: 1
    command:
      ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Then run,
```bash
docker-compose up
```

## Docker Image

The Docker image for this project is available on Docker Hub. You can pull the image using the following command:

```bash
docker pull karabairak/aging:latest
```

## Usage
You can access the endpoints and send the requests using curl or Postman.
```bash
curl -X POST "http://34.91.249.118:80/api/aging/sam_model" -H "Content-Type: application/json" -d '{"base64_img": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxISEBU..."}'
```

## Demo Scripts
* demo.py: Converts a given image into a Base64 encoded string.
* base642img.py: Converts the response to the images with age transformations.




