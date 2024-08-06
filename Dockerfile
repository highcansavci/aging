# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install required system packages and tools
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    g++ \
    wget \
    curl \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY . /app

# Install PyTorch with CUDA 12.1
RUN pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for downloaded files
RUN mkdir -p pretrained_models

# Download the required files
RUN gdown "https://drive.google.com/u/0/uc?id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC&export=download" -O pretrained_models/sam_ffhq_aging.pt && \
    curl -o pretrained_models/shape_predictor_68_face_landmarks.dat https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat

# Expose the port that the app runs on
EXPOSE 8000

# Define the command to run the application, adjusting the path to main.py
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
