FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

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

# Copy the repository
WORKDIR /app
COPY . /app

# Run setup script (this installs python dependencies, patches submodule, and downloads checkpoints)
# Note: We run this during build to bake everything into the image.
# Ideally, checkpoints should be mounted, but the setup script downloads them.
# We can modify setup_env.py to skip download if files exist, or we can let it run.
# For a robust docker image, baking code is good, but checkpoints are large.
# We will let setup_env.py run.
# IMPORTANT: setup_env.py requires MASt3R-SLAM submodule to be present.
# Since we COPY . /app, the empty directory might be there, but not content if not initialized.
# The setup_env.py has logic to `git submodule update --init`, but inside a container, git auth might fail if using SSH.
# Public HTTPS URLs should work.

RUN python3 setup_env.py

# Default command
CMD ["python3", "run_slam.py", "--all"]
