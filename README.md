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

1.  **Prerequisites:**
    *   Ensure you have Python 3.12.10 installed. You can download it from [python.org](https://www.python.org/).
    *   Ensure you have Pip installed and updated (usually comes with Python).

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Create a Virtual Environment (Recommended):**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
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

This project includes a `Dockerfile` to build a containerized environment with all dependencies pre-installed.

**1. Build the Docker Image:**

Navigate to the root directory of the project (where the `Dockerfile` is located) and run:
```bash
docker build -t 3d-slam-system .
```
This command builds a Docker image tagged as `3d-slam-system`.

**2. Run the Docker Container:**

The default command for the container is `python scripts/run_pointcloud_generation.py`.

*   **Basic Run (No Camera, No GUI, Ephemeral Data):**
    ```bash
    docker run -it --rm 3d-slam-system
    ```
    This will run the main script. However, without camera access, it will likely fail if the script expects a live camera. For scripts that process existing files and don't require live input or GUI, this might suffice.

*   **With Camera Access (Linux):**
    To allow the container to access your webcam, you need to pass the device. `/dev/video0` is commonly the first webcam.
    ```bash
    docker run -it --rm --device=/dev/video0 3d-slam-system
    ```
    If you have multiple cameras, find the correct device (e.g., `/dev/video1`, etc.).

*   **With GUI Display (Linux - X11 Forwarding):**
    The main script `run_pointcloud_generation.py` uses Open3D for visualization, which requires a display. To enable GUI applications from within the container to display on your host's X server:
    ```bash
    docker run -it --rm \
        --device=/dev/video0 \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        3d-slam-system
    ```
    Note: X11 forwarding can have security implications and might require `xhost +local:docker` or similar commands on the host, depending on your X server configuration. This setup is primarily for Linux hosts. GUI forwarding on macOS or Windows with Docker Desktop might require different setups (e.g., using an XQuartz for macOS or VcXsrv for Windows).

*   **Persisting Data (Maps and PyTorch Hub Cache):**
    *   **Maps & Calibration Data:** To save generated maps and camera calibration files outside the container (so they are not lost when the container stops), mount the local `data` directory:
        ```bash
        -v $(pwd)/data:/app/data
        ```
        Saved maps (e.g., `generated_map_tsdf.ply`) and `camera_calibration.yaml` will appear in your local `data` directory.
    *   **PyTorch Hub Model Caching:** MiDaS models are downloaded by PyTorch Hub and cached within the container (typically in `/root/.cache/torch/hub`, as the container often runs as root). If you want to persist this cache across container *rebuilds* or different containers, you can mount your host's PyTorch Hub cache directory to the container's cache location. This is optional but can save re-downloading models frequently.
        ```bash
        # Example for Linux/macOS hosts:
        -v ~/.cache/torch/hub:/root/.cache/torch/hub
        ```
        The local `models/` directory is not used by the PyTorch MiDaS implementation for caching.

    **Example `docker run` with GUI, Camera, and Persistent Data/Cache:**
    ```bash
    docker run -it --rm \
        --device=/dev/video0 \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $(pwd)/data:/app/data \
        -v ~/.cache/torch/hub:/root/.cache/torch/hub \
        3d-slam-system
    ```

*   **Running Other Scripts:**
    You can override the default CMD to run other scripts. For example, to run the interactive calibration assistant:
    ```bash
    docker run -it --rm \
        --device=/dev/video0 \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $(pwd)/data:/app/data \
        3d-slam-system python scripts/calibration_assistant.py
    ```
    (Ensure your `data/calibration_images/` directory exists on the host if you plan to save images there via the assistant, though the assistant saves directly to the mounted `/app/data`).

**Important Considerations for Docker:**

*   **Camera Device:** The `--device` flag is Linux-specific. Docker Desktop on Windows and macOS handles camera access differently, often transparently if the application requests it and permissions are granted. You might not need the `--device` flag on these platforms, but it depends on the Docker version and configuration.
*   **GUI on macOS/Windows:** As mentioned, X11 forwarding is more complex. For macOS, XQuartz is needed. For Windows, VcXsrv or WSLg (with Docker in WSL2) can be used. Simpler alternatives for visualization might involve saving output files and viewing them on the host.
*   **Performance:** Docker might introduce some performance overhead, especially for I/O operations or if running on a non-native platform (e.g., x86 image on an ARM Mac via Rosetta 2).
*   **Resource Limits:** By default, Docker containers have access to a portion of the host's resources. For computationally intensive tasks like SLAM, ensure Docker is configured with sufficient CPU and memory.

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

```
