FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set TORCH_CUDA_ARCH_LIST to ensure extensions build for common architectures
# even if no GPU is present during build time.
# 7.5 (Turing), 8.0 (Ampere), 8.6 (Ampere), 8.9 (Ada), 9.0 (Hopper)
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install build dependencies
# requests is needed for setup_env.py
# numpy/setuptools often needed for building extensions
RUN python3 -m pip install requests numpy setuptools opencv-python-headless

# Configure git to allow operations on /app (needed for submodule update)
RUN git config --global --add safe.directory /app
RUN git config --global --add safe.directory /app/MASt3R-SLAM

# Copy the repository
WORKDIR /app
COPY . /app

# Run setup script (this installs python dependencies, patches submodule, and downloads checkpoints)
# We rely on the script to handle the heavy lifting.
RUN python3 setup_env.py

# Default command
CMD ["python3", "run_slam.py", "--all"]
