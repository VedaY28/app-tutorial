# Use NVIDIA PyTorch container with CUDA
FROM nvcr.io/nvidia/pytorch:24.06-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    mosquitto-clients \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your local code into the container (excluding large videos)
COPY main.py firedetect-11s.pt ./

# Patch main.py to fix Ultralytics speed key
RUN sed -i 's/result.speed\[\"dataloading\"\]/result.speed.get(\"preprocess\", 0)/g' /app/main.py

# Default command
ENTRYPOINT ["python3", "main.py"]
