# Real-time Monocular SLAM and 3D Reconstruction

This project implements a real-time monocular Simultaneous Localization and Mapping (SLAM) system with 3D reconstruction capabilities. It utilizes a single camera to estimate its motion (visual odometry), perceive depth in the scene, and build a 3D map of the environment using TSDF (Truncated Signed Distance Function) integration.

## Features

*   **Monocular Camera Input:** Processes video feed from a standard webcam or video file.
*   **MiDaS Depth Estimation (PyTorch Hub):** Employs pre-trained MiDaS v3.1 models (e.g., DPT_Hybrid, MiDaS_small) from PyTorch Hub (`intel-isl/MiDaS`) to estimate depth from monocular images.
*   **ORB-based Visual Odometry (VO):** Tracks camera pose (rotation and translation) by detecting and matching ORB features between frames.
*   **TSDF Dense Reconstruction:** Builds a dense 3D map of the environment using Open3D's `ScalableTSDFVolume`. This allows for robust fusion of depth data from multiple viewpoints.
*   **Point Cloud & Mesh Generation:** Can extract a global point cloud or a 3D mesh from the TSDF volume.
*   **Map Saving/Loading:**
    *   Saves the reconstructed point cloud map to a `.ply` file.
    *   Loads a `.ply` point cloud file for visualization (note: live TSDF reconstruction will restart if a map is loaded this way).
*   **Camera Calibration:** Includes scripts (`scripts/calibrate_camera.py` and `scripts/calibration_assistant.py`) to calibrate the camera using checkerboard images, saving parameters to a YAML file. The interactive `calibration_assistant.py` features live checkerboard detection, zone coverage guidance, and an auto-capture mode (toggle with 'a' key). Default internal corners for calibration set to 12x8.
*   **Modular Design:** Code is organized into modules for camera handling, depth estimation, VO, and reconstruction.

## Project Structure

```
.
├── data/                     # Directory for input/output data
│   ├── calibration_images/   # Sample images for camera calibration (user provides their own)
│   ├── camera_calibration.yaml # Output of the calibration script (example provided)
│   └── generated_map_tsdf.ply  # Example of a saved map (ignored by Git)
├── environment.yml           # Conda environment definition
├── models/                   # Directory for user-saved models or other non-PyTorch Hub models. MiDaS models from PyTorch Hub are cached by PyTorch (see Setup).
├── README.md                 # This file
├── scripts/                  # Python scripts to run parts of or the full application
│   ├── calibrate_camera.py   # Script for camera calibration (manual image placement)
│   ├── calibration_assistant.py # Interactive script for camera calibration image collection & processing
│   ├── run_pointcloud_generation.py # Main script for full SLAM & reconstruction
│   ├── run_vo.py             # Script to run only Visual Odometry
│   ├── view_camera.py        # Script to test camera input
│   └── view_depth.py         # Script to test depth estimation
├── src/                      # Source code for the SLAM system modules
│   ├── camera/               # MonocularCamera class
│   ├── depth_estimation/     # MiDaSDepthEstimator class
│   ├── reconstruction/       # PointCloudMapper class (with TSDF)
│   ├── slam/                 # VisualOdometry class
│   └── utils/                # CameraParams class and other utilities
├── tests/                    # Unit tests
│   ├── __init__.py
│   └── test_camera_params.py
└── .gitignore                # Specifies intentionally untracked files
```

## Setup Instructions

It is recommended to use Python 3.11 to set up your local development environment, as this version has good compatibility with all project dependencies, including `open3d` on macOS.

1.  **Prerequisites:**
    *   Ensure you have Python 3.11 installed. You can download it from [python.org](https://www.python.org/).
    *   Ensure you have Pip installed and updated (usually comes with Python).

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Create a Virtual Environment (Recommended):**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3.11 -m venv .venv
    ```
    Activate the virtual environment:
    *   On Windows:
        ```bash
        .\.venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```

4.  **Install Dependencies:**
    Install all necessary Python dependencies using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Camera Connection:**
    *   Ensure you have a webcam connected to your system if you intend to use live camera input.
    *   The default camera ID used is `0`. If you have multiple cameras, you might need to change this ID in the scripts (e.g., in `MonocularCamera(0)` calls).

6.  **Pre-cache MiDaS Model Weights (PyTorch Hub):**
    The system uses MiDaS v3.1 models (e.g., DPT_Hybrid, MiDaS_small) via PyTorch Hub (`intel-isl/MiDaS`). PyTorch Hub automatically handles the download and caching of these models (typically in `~/.cache/torch/hub/` on Linux/macOS or `C:\Users\<username>\.cache\torch\hub\` on Windows).

    The main application scripts (`run_pointcloud_generation.py`, `view_depth.py`) will trigger this download on their first run if the selected model is not already cached. To pre-cache models, especially if you plan to work offline or want to ensure all desired models are available without delay during application startup, use the `scripts/download_midas_model.py` script.

    **Usage of `scripts/download_midas_model.py`:**
    ```bash
    # Ensure the default 'hybrid' (DPT_Hybrid) MiDaS model is cached
    python scripts/download_midas_model.py

    # Ensure the 'small' (MiDaS_small) model is cached
    python scripts/download_midas_model.py --model_type small

    # Ensure the 'large_beit_512' (DPT_BEiT_L_512) model is cached
    python scripts/download_midas_model.py --model_type large_beit_512

    # Ensure all defined models ('small', 'hybrid', 'large_beit_512') are cached
    python scripts/download_midas_model.py --model_type all
    ```
    Running this script helps populate the PyTorch Hub cache, making subsequent application runs faster if the models weren't already present. The local `models/` directory in the project is not used for caching these PyTorch Hub MiDaS models.

## Running with Docker

This project includes a `Dockerfile` to build a containerized environment with all dependencies pre-installed. The Docker image uses **Python 3.11**, aligning with the recommended local setup for better compatibility (especially for `open3d` on macOS).

**1. Build the Docker Image:**

Navigate to the root directory of the project (where the `Dockerfile` is located) and run:
```bash
docker build -t 3d-slam-system .
```
This command builds a Docker image tagged as `3d-slam-system`.

**For macOS M-series (M1/M2/M3) users:**
The Docker image is built on a multi-architecture base image (`python:3.11-slim-bookworm`) that supports `arm64`. Docker on your M-series Mac should automatically pull and use the correct architecture, so you typically **do not need to specify a `--platform` flag**.

**2. Run the Docker Container:**

The default command for the container is `python scripts/run_pointcloud_generation.py`.

*   **General Run Command Structure:**
    ```bash
    docker run -it --rm \
        [--device=/dev/video0] \
        [-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix] \
        -v $(pwd)/data:/app/data \
        -v ~/.cache/torch/hub:/root/.cache/torch/hub \
        3d-slam-system [python scripts/your_script.py --args]
    ```
    *   `--device=/dev/video0`: (Linux specific) Grants camera access. May not be needed on Docker Desktop for Mac/Windows if camera access is handled by the desktop application.
    *   `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`: (Linux specific) For GUI forwarding. See notes below for macOS/Windows.
    *   `-v $(pwd)/data:/app/data`: **Crucial for persisting data.** Mounts your local `./data` directory (containing `camera_calibration.yaml`, saved maps, calibration images) into the container's `/app/data`. **Ensure your local `data` directory exists.**
    *   `-v ~/.cache/torch/hub:/root/.cache/torch/hub`: **Recommended for persisting MiDaS models.** Mounts your host's PyTorch Hub cache. This prevents re-downloading models every time you run or rebuild the container. The container runs as root, so its cache is at `/root/.cache/torch/hub`.
    *   `3d-slam-system`: The name of the image you built.
    *   `[python scripts/your_script.py --args]`: Optional. If not provided, runs the default CMD (`python scripts/run_pointcloud_generation.py`).

*   **Workflow for Camera Calibration and Running Main Application:**

    1.  **Create `data` directory on host:** If it doesn't exist, create it: `mkdir data`.
    2.  **Run Calibration Assistant (if `camera_calibration.yaml` is missing or needs update):**
        This script will save `camera_calibration.yaml` and calibration images into the mounted `/app/data` directory (which is your local `./data` directory).
        ```bash
        # Linux example with camera and GUI
        docker run -it --rm \
            --device=/dev/video0 \
            -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v $(pwd)/data:/app/data \
            -v ~/.cache/torch/hub:/root/.cache/torch/hub \
            3d-slam-system python scripts/calibration_assistant.py
        
        # macOS/Windows (Docker Desktop might handle camera/GUI differently - adapt if needed)
        # Basic command, assuming Docker Desktop handles camera/GUI:
        docker run -it --rm \
            -v $(pwd)/data:/app/data \
            -v ~/.cache/torch/hub:/root/.cache/torch/hub \
            3d-slam-system python scripts/calibration_assistant.py
        ```
        *Note: The `Dockerfile` copies a `data/camera_calibration.yaml` if present during the build. Running the assistant ensures you use your specific, potentially updated, calibration.*

    3.  **Run Main SLAM Application:**
        This will use the `camera_calibration.yaml` from your mounted `./data` directory.
        ```bash
        # Linux example with camera and GUI
        docker run -it --rm \
            --device=/dev/video0 \
            -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v $(pwd)/data:/app/data \
            -v ~/.cache/torch/hub:/root/.cache/torch/hub \
            3d-slam-system
            
        # macOS/Windows (Docker Desktop - adapt if needed)
        docker run -it --rm \
            -v $(pwd)/data:/app/data \
            -v ~/.cache/torch/hub:/root/.cache/torch/hub \
            3d-slam-system
        ```

**Important Considerations for Docker:**

*   **Camera Device Access:** The `--device=/dev/video0` flag is common for Linux. Docker Desktop on Windows and macOS often handles camera access more transparently if the application requests it and permissions are granted. You might not need this flag, or it might require different configuration through Docker Desktop settings.
*   **GUI on macOS/Windows:** X11 forwarding (`-e DISPLAY -v /tmp/.X11-unix`) is primarily for Linux.
    *   **macOS:** Requires XQuartz. You might need to set `DISPLAY` to `host.docker.internal:0` or similar and configure XQuartz to allow network connections.
    *   **Windows:** Requires an X server like VcXsrv or can use WSLg if running Docker within WSL2.
    *   For simplicity, you might run scripts without direct GUI visualization in Docker and save outputs (like point clouds) to the mounted `/app/data` to view them on your host system.
*   **Performance:** Docker might introduce some performance overhead, especially for I/O operations or if running an x86_64 image on an ARM Mac via Rosetta 2 (though this `Dockerfile` aims for native ARM64 on M-series Macs).
*   **Resource Limits:** Ensure Docker is configured with sufficient CPU and memory, especially for computationally intensive tasks like SLAM. Access Docker Desktop's "Resources" settings to adjust.

## Running the Application

### 1. Camera Calibration (Recommended First Step)

Accurate camera intrinsic parameters are crucial for SLAM. This project provides two scripts for calibration:

*   **`scripts/calibrate_camera.py` (Manual Image Placement):**
    *   **Prepare Checkerboard Pattern:** Print or obtain a physical checkerboard. The script defaults to `CHECKERBOARD_INTERNAL_CORNERS = (12, 8)` (meaning 13x9 squares) and `SQUARE_SIZE_MM = 20.0`. If your checkerboard differs, update these constants in the script.
    *   **Capture Calibration Images:** Take 15-20 clear images of the checkerboard from various angles and distances. Place these images (e.g., `.png`, `.jpg`) into the `data/calibration_images/` directory.
    *   **Run Calibration Script:**
        ```bash
        python scripts/calibrate_camera.py
        ```
*   **`scripts/calibration_assistant.py` (Interactive Assistant):**
    *   This script provides an interactive way to collect calibration images using a live camera feed.
    *   It features live checkerboard detection, on-screen guidance for covering different zones of the camera view, and an auto-capture mode (toggle with the 'a' key) for convenience.
    *   It also defaults to 12x8 internal checkerboard corners but allows you to input custom dimensions.
    *   **Run Interactive Assistant:**
        ```bash
        python scripts/calibration_assistant.py
        ```
        Follow the on-screen prompts to capture images, which will be saved to `data/calibration_images/`. After collection, it proceeds to calibration.

Both scripts save the calibration results to `data/camera_calibration.yaml`, which is then used by the main SLAM application.

### 2. Running the Full SLAM and 3D Reconstruction Pipeline

This is the main application script.

*   **To run with live camera feed and start a new map:**
    ```bash
    python scripts/run_pointcloud_generation.py
    ```
*   **To load an existing point cloud map for visualization (live TSDF will restart):**
    ```bash
    python scripts/run_pointcloud_generation.py --load_map data/your_map.ply
    ```
    (Replace `data/your_map.ply` with the actual path to your map file).

### 3. Running Other Utility Scripts

These scripts are useful for testing individual components:

*   **Test Camera Input:**
    ```bash
    python scripts/view_camera.py
    ```
*   **Test Depth Estimation:**
    ```bash
    python scripts/view_depth.py
    ```
*   **Test Visual Odometry Only:**
    ```bash
    python scripts/run_vo.py
    ```

## Keyboard Controls (for `run_pointcloud_generation.py`)

When the Open3D visualizer window is active:

*   `q`: Quit the application.
*   `k`: Save the current map (TSDF-extracted point cloud) to `data/generated_map_tsdf.ply` (or the path defined by `DEFAULT_SAVE_PATH` in the script).
*   `l`: Load a map from `data/generated_map_tsdf.ply` (or `DEFAULT_LOAD_PATH`). This is primarily for visualization; the live TSDF volume will be reset, and new mapping will start fresh.
*   `m`: Extract and view the 3D mesh from the current TSDF volume in a new window.

## Dependencies

Key Python libraries used:

*   **OpenCV (`cv2`):** For camera handling, image processing, feature detection/matching, and VO core algorithms.
*   **Open3D (`open3d`):** For 3D data structures (point clouds, meshes), TSDF volume integration, and 3D visualization.
*   **PyTorch (`torch`), TorchVision (`torchvision`), TIMM (`timm`):** Used for loading and running the MiDaS v3.1 depth estimation models from PyTorch Hub.
*   **NumPy:** For numerical operations.
*   **PyYAML:** For saving and loading camera calibration data.
*   **Pandas:** For potential data manipulation tasks (though not heavily used in the core SLAM logic).
*   **tqdm:** For displaying progress bars, especially during model downloads.

All required Python packages are listed in `requirements.txt` and can be installed using pip.

## Known Issues / Limitations

*   **Monocular VO Scale Ambiguity:** The translation component from monocular VO has an arbitrary scale. While the MiDaS depth provides some scale information, the consistency and accuracy of this scale throughout the trajectory can vary.
*   **Depth Scale Factor:** The `DEPTH_OUTPUT_SCALE_FACTOR` in `scripts/run_pointcloud_generation.py` is a crucial parameter for converting MiDaS relative depth to metric depth. This factor may need empirical tuning based on the camera, scene, and MiDaS model version for optimal reconstruction.
*   **VO Robustness:** Feature-based VO can be sensitive to:
    *   Fast camera movements or rotations.
    *   Poorly textured environments.
    *   Significant lighting changes.
*   **Loop Closure:** This implementation does not include loop closure, so drift can accumulate over long trajectories.
*   **Real-time Performance:** Depth estimation (especially larger MiDaS models) can be computationally intensive. Performance depends on the CPU/GPU capabilities.

## Future Work (Possible Enhancements)

*   Implement loop closure to reduce drift.
*   Bundle adjustment for pose graph optimization.
*   More sophisticated depth map scaling or fusion.
*   Integration with IMU data for improved robustness.
*   UI for easier parameter tuning and control.

