FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    ffmpeg \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libx11-6 \
    libglib2.0-0 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip for python3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install build dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.4.1 with CUDA 12.4
RUN python3 -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# Configure git
RUN git config --global --add safe.directory /app
RUN git config --global --add safe.directory /app/MASt3R-SLAM

WORKDIR /app
COPY . /app

# Run setup script
RUN python3 setup_env.py

# Expose ports for Rerun (9876) and Gradio (7860)
EXPOSE 9876 7860

CMD ["python3", "run_slam.py"]
