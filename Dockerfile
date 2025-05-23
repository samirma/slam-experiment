# 1. Base Image
FROM python:3.11-slim-bookworm

# 2. Set Working Directory
WORKDIR /app

# 3. Install system-level dependencies
# Ensure non-interactive frontend for apt-get
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    # Matplotlib dependencies (headless)
    libfreetype6-dev \
    libpng-dev \
    # Open3D dependencies (headless)
    # libgl1-mesa-glx is already listed for OpenCV
    libegl1-mesa \
    libosmesa6 \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements.txt
COPY requirements.txt /app/requirements.txt

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy src directory
COPY src/ /app/src/

# 7. Copy scripts directory
COPY scripts/ /app/scripts/

# 8. Copy camera_calibration.yaml
# The COPY instruction will create the /app/data/ directory if it doesn't exist.
COPY data/camera_calibration.yaml /app/data/camera_calibration.yaml

# 10. Create a placeholder for models directory (for potential volume mounting)
# This directory will be populated by midas.py on first run if not mounted.
RUN mkdir -p /app/models

# 9. Define default CMD
# Update this if a different script or arguments are preferred for default execution
CMD ["python", "scripts/run_pointcloud_generation.py"]
