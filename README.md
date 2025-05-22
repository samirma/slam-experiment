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

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create Conda Environment:**
    Ensure you have Anaconda or Miniconda installed. Then, create the environment using the provided file:
    ```bash
    conda env create -f environment.yml
    ```
    This will install all necessary Python dependencies, including OpenCV, Open3D, TensorFlow, and TensorFlow Hub.

3.  **Activate Conda Environment:**
    ```bash
    conda activate 3d-slam-system
    ```
    (The environment name is defined in `environment.yml`, ensure it matches or use the correct name if you changed it).

4.  **Camera Connection:**
    *   Ensure you have a webcam connected to your system if you intend to use live camera input.
    *   The default camera ID used is `0`. If you have multiple cameras, you might need to change this ID in the scripts (e.g., in `MonocularCamera(0)` calls).

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

These are all included in the `environment.yml` file.

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
