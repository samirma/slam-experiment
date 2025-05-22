# Real-time Monocular SLAM and 3D Reconstruction

This project implements a real-time monocular Simultaneous Localization and Mapping (SLAM) system with 3D reconstruction capabilities. It utilizes a single camera to estimate its motion (visual odometry), perceive depth in the scene, and build a 3D map of the environment using TSDF (Truncated Signed Distance Function) integration.

## Features

*   **Monocular Camera Input:** Processes video feed from a standard webcam or video file.
*   **MiDaS Depth Estimation:** Employs a pre-trained MiDaS model (from TensorFlow Hub) to estimate depth from monocular images.
*   **ORB-based Visual Odometry (VO):** Tracks camera pose (rotation and translation) by detecting and matching ORB features between frames.
*   **TSDF Dense Reconstruction:** Builds a dense 3D map of the environment using Open3D's `ScalableTSDFVolume`. This allows for robust fusion of depth data from multiple viewpoints.
*   **Point Cloud & Mesh Generation:** Can extract a global point cloud or a 3D mesh from the TSDF volume.
*   **Map Saving/Loading:**
    *   Saves the reconstructed point cloud map to a `.ply` file.
    *   Loads a `.ply` point cloud file for visualization (note: live TSDF reconstruction will restart if a map is loaded this way).
*   **Camera Calibration:** Includes a script to calibrate the camera using checkerboard images, saving parameters to a YAML file.
*   **Modular Design:** Code is organized into modules for camera handling, depth estimation, VO, and reconstruction.

## Project Structure

```
.
├── data/                     # Directory for input/output data
│   ├── calibration_images/   # Sample images for camera calibration (user provides their own)
│   ├── camera_calibration.yaml # Output of the calibration script (example provided)
│   └── generated_map_tsdf.ply  # Example of a saved map (ignored by Git)
├── environment.yml           # Conda environment definition
├── models/                   # (Currently unused, but could store local TF models if not using TF Hub)
├── README.md                 # This file
├── scripts/                  # Python scripts to run parts of or the full application
│   ├── calibrate_camera.py   # Script for camera calibration
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

*   **Persisting Data (Maps and Models):**
    To save generated maps or downloaded MiDaS models outside the container (so they are not lost when the container stops), use Docker volumes:
    ```bash
    docker run -it --rm \
        --device=/dev/video0 \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/models:/app/models \
        3d-slam-system
    ```
    This mounts the local `./data` directory to `/app/data` in the container and `./models` to `/app/models`.
    *   Saved maps (e.g., `generated_map_tsdf.ply`) will appear in your local `data` directory.
    *   Downloaded MiDaS models will be cached in your local `models` directory, preventing re-downloads on subsequent runs.

*   **Running Other Scripts:**
    You can override the default CMD to run other scripts. For example, to run the camera calibration script:
    ```bash
    docker run -it --rm \
        -v $(pwd)/data:/app/data \
        3d-slam-system python scripts/calibrate_camera.py
    ```
    (Ensure your `data/calibration_images/` directory is populated on the host for this example).

**Important Considerations for Docker:**

*   **Camera Device:** The `--device` flag is Linux-specific. Docker Desktop on Windows and macOS handles camera access differently, often transparently if the application requests it and permissions are granted. You might not need the `--device` flag on these platforms, but it depends on the Docker version and configuration.
*   **GUI on macOS/Windows:** As mentioned, X11 forwarding is more complex. For macOS, XQuartz is needed. For Windows, VcXsrv or WSLg (with Docker in WSL2) can be used. Simpler alternatives for visualization might involve saving output files and viewing them on the host.
*   **Performance:** Docker might introduce some performance overhead, especially for I/O operations or if running on a non-native platform (e.g., x86 image on an ARM Mac via Rosetta 2).
*   **Resource Limits:** By default, Docker containers have access to a portion of the host's resources. For computationally intensive tasks like SLAM, ensure Docker is configured with sufficient CPU and memory.

## Running the Application

### 1. Camera Calibration (Recommended First Step)

The accuracy of the SLAM system, especially visual odometry and 3D reconstruction, heavily depends on accurate camera intrinsic parameters.

*   **Prepare Checkerboard Pattern:**
    *   Print or obtain a physical checkerboard pattern. The script defaults to `CHECKERBOARD_INTERNAL_CORNERS = (9, 6)` (meaning 10x7 squares) and `SQUARE_SIZE_MM = 20.0`.
    *   If your checkerboard is different, update these constants at the top of `scripts/calibrate_camera.py`.
*   **Capture Calibration Images:**
    *   Take 15-20 clear images of the checkerboard from various angles and distances, ensuring the entire board is visible.
    *   Place these images (e.g., `.png`, `.jpg`) into the `data/calibration_images/` directory. (Example images might be provided, but using your own camera's images is crucial).
*   **Run Calibration Script:**
    ```bash
    python scripts/calibrate_camera.py
    ```
    The script will process the images and save the calibration results to `data/camera_calibration.yaml`. This file will be automatically used by other scripts.

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
*   **TensorFlow & TensorFlow Hub (`tensorflow`, `tensorflow_hub`):** For loading and running the MiDaS depth estimation model.
*   **NumPy:** For numerical operations.
*   **PyYAML:** For saving and loading camera calibration data.
*   **Pandas:** For potential data manipulation tasks (though not heavily used in the core SLAM logic).

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
