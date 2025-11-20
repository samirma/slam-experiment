FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set TORCH_CUDA_ARCH_LIST to ensure extensions build for common architectures
# even if no GPU is present during build time.
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0"

# Install system dependencies
# Added libgl1-mesa-dri, mesa-utils, libx11-6 for X11/OpenGL support (visualization)
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libx11-6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install build dependencies
RUN python3 -m pip install requests numpy setuptools opencv-python-headless

# Install PyTorch (needed for compiling extensions like lietorch and MASt3R-SLAM backend)
# Using CUDA 12.1 build
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Configure git to allow operations on /app (needed for submodule update)
RUN git config --global --add safe.directory /app
RUN git config --global --add safe.directory /app/MASt3R-SLAM

# Copy the repository
WORKDIR /app
COPY . /app

# Run setup script (this installs python dependencies, patches submodule, and downloads checkpoints)
RUN python3 setup_env.py

# Default command
CMD ["python3", "run_slam.py", "--all"]
