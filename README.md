# MASt3R-SLAM for Webcams

This project provides a wrapper around [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) to enable real-time 3D reconstruction using standard webcams.

## Prerequisites

*   **NVIDIA GPU**: Required for MASt3R-SLAM (CUDA dependencies).
*   **Ubuntu** (Recommended) or Linux with NVIDIA Container Toolkit.
*   **Webcams**: One or more monocular webcams.

## Setup (Native)

1.  **Clone the repository:**
    ```bash
    git clone --recursive <repo_url>
    cd <repo_name>
    ```

2.  **Run the setup script:**
    This script checks for CUDA, initializes submodules, installs dependencies, and downloads necessary checkpoints.
    ```bash
    python3 setup_env.py
    ```

3.  **Run SLAM:**
    To run on all available cameras:
    ```bash
    python3 run_slam.py --all
    ```
    To run on a specific camera (e.g., camera 0):
    ```bash
    python3 run_slam.py --cam-id 0
    ```

## Docker Setup (Ubuntu + NVIDIA)

To avoid messing with system dependencies, you can use Docker.

1.  **Install NVIDIA Container Toolkit:**
    Follow instructions at: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

2.  **Build the image:**
    ```bash
    docker build -t mast3r-slam-webcam .
    ```

3.  **Run the container:**
    You need to pass the GPU and the webcam devices to the container.
    ```bash
    docker run --gpus all --device /dev/video0:/dev/video0 --device /dev/video1:/dev/video1 -it mast3r-slam-webcam
    ```
    *(Adjust `/dev/videoX` mapping as needed for your cameras).*

## macOS Support

**Status: Not Supported / Experimental**

MASt3R-SLAM relies heavily on **CUDA** custom kernels (in `curope`, `croco`, `lietorch`, and the backend). These kernels are compiled specifically for NVIDIA GPUs.

*   **Apple Silicon (M1/M2/M3):** Not supported. There is no MPS (Metal) port of the custom CUDA kernels.
*   **Intel Mac:** Not supported unless you use an external NVIDIA GPU (e-GPU) and pass it through to a Linux VM/Docker, which is a complex setup.

If you attempt to build the Docker image on macOS (e.g., via Docker Desktop), it will fail or run in emulation mode (extremely slow) and crash when attempting to load CUDA libraries.

For development (editing code only), you can build the image without the final execution step, but the Python code will not run successfully.
