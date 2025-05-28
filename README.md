# SLAM Experiment Project

This project implements a basic Simultaneous Localization and Mapping (SLAM) pipeline using Python and OpenCV. It includes modules for camera calibration, feature detection and matching, and 2-view structure from motion (SfM) to estimate camera pose and triangulate 3D points. The system now supports multiple cameras, per-camera calibration, real-time 3D reconstruction visualization, and display of the estimated camera pose in the 3D view.

## Project Structure

```
slam-experiment/
├── requirements.txt
├── README.md
├── src/
│   ├── calibration.py    # Handles camera calibration
│   ├── feature_utils.py  # Utilities for feature detection and matching
│   ├── main.py           # Main application script
│   └── sfm.py            # Functions for Structure from Motion (pose estimation, triangulation)
└── calibration_data/     # Directory to save calibration images and parameters (created automatically)
```

## Features

* **Multi-Camera Support**: Users can select from a list of available cameras connected to the system.
* **Per-Camera Calibration**: Captures checkerboard images and calculates camera intrinsic matrix and distortion coefficients for the selected camera. Calibration data is saved and loaded specific to each camera index (e.g., `camera_params_idx0.npz`).
* **Feature Detection**: Uses ORB detector to find keypoints in images.
* **Feature Matching**: Matches features between consecutive frames using a brute-force matcher with ratio test.
* **Pose Estimation**: Estimates camera rotation and translation between two views using the 5-point algorithm for the Essential Matrix and RANSAC. The estimated pose is accumulated over frames.
* **3D Triangulation**: Reconstructs 3D points from matched 2D keypoints and estimated poses.
* **Live Feed**: Shows live webcam feed with undistorted images, detected keypoints, and feature matches.
* **Real-time 3D Reconstruction Visualization**: Displays the triangulated 3D point cloud in a dedicated 3D window using the Open3D library.
* **Camera Pose Visualization**: Shows the estimated camera pose trajectory (represented by coordinate axes) in the Open3D window, including the current camera position and the initial world origin.

## Setup and Installation

Follow these steps to set up your environment and run the project:

### 1. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

**On macOS and Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt, indicating that the virtual environment is active.

### 2. Install Dependencies

Once the virtual environment is activated, install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
The primary dependencies are `numpy`, `opencv-python` (for core computer vision tasks), and `open3d` (for 3D visualization). The `requirements.txt` file has been updated to include `open3d`.

## Running the Application

To run the SLAM experiment, execute the `main.py` script from the root directory of the project:

```bash
python src/main.py
```

### Workflow:

1.  **Camera Selection**:
    *   The application starts by probing for available cameras (up to a predefined maximum to check).
    *   A list of detected cameras will be printed to the console. Each camera is listed with:
        *   Its **Camera ID** (numerical index, e.g., `Camera ID: 0`).
        *   Its current **Calibration Status** (e.g., `(Calibration: Yes)` or `(Calibration: No)`).
    *   Example output:
        ```
        Available Cameras:
        - Camera ID: 0 (Calibration: Yes)
        - Camera ID: 1 (Calibration: No)
        ```
    *   You will then be prompted to enter the ID of the camera you wish to use from the displayed list (e.g., `Enter the ID of the camera you want to use (options: [0, 1]):`).
2.  **Calibration Data Check (Per Camera)**:
    *   The application attempts to load existing camera calibration data for the *selected* camera ID (e.g., from `calibration_data/camera_params_idx0.npz` if camera ID `0` was selected).
3.  **Camera Calibration (if needed for the selected camera)**:
    *   If calibration data for the selected camera is not found or fails to load, the camera calibration process will start using the selected camera.
    *   A live feed from your selected webcam will be displayed in the "Calibration Feed" window.
    *   Position a checkerboard (default internal corners: 12x8) in front of the camera.
    *   Images are captured automatically when a checkerboard is detected and sufficient time has passed since the last capture. The system requires a minimum of 5 images (configurable by `MIN_CALIBRATION_IMAGES`) and aims to capture up to 15 images (configurable by `MAX_IMAGES`).
    *   Captured images (grayscale) are saved in the `calibration_data/` directory (e.g., `calib_img_1.png`).
    *   Press **'q'** to finish capturing images and proceed to calibration (if enough images are captured) or to quit the capture mode.
    *   Press **ESC** to discard all captures and exit the calibration image capture process.
    *   If calibration is successful, the parameters (`camera_matrix`, `dist_coeffs`, `frame_size`, `reprojection_error`) are saved to a camera-specific file in the `calibration_data/` directory (e.g., `camera_params_idx0.npz`).
4.  **Feature Detection, Matching, and 3D Reconstruction Loop**:
    *   Once the selected camera is calibrated (or existing data is loaded), the application starts the main SfM loop.
    *   **Live Feed Windows**:
        *   "Live Feed with Keypoints": Shows the undistorted live video from the selected camera, with detected ORB keypoints overlaid.
        *   "Feature Matches": Displays matches between features from the current and previous frames.
    *   **3D Visualization Window ("3D Reconstruction")**:
        *   An Open3D window titled "3D Reconstruction" will open to display the 3D scene.
        *   This window shows:
            *   A large coordinate system widget representing the world origin.
            *   A smaller coordinate system widget representing the current estimated pose of the camera, updated in real-time.
            *   A point cloud representing the triangulated 3D points from the scene.
        *   You can interact with this 3D scene using standard Open3D mouse controls (e.g., left-click and drag to rotate, right-click and drag or scroll wheel to zoom, middle-click and drag to pan).
    *   **Console Output**: Information about pose estimation success/failure and number of reconstructed points will be printed to the console.
5.  **Exiting**:
    *   Press **'q'** in any of the OpenCV display windows (Live Feed, Matches) to quit the main loop.
    *   Closing the Open3D "3D Reconstruction" window will also terminate the application.

## Key Files and Modules

* **`src/main.py`**: The main entry point. Handles camera selection, orchestrates the per-camera calibration process, manages the main SfM loop, and integrates 3D visualization.
* **`src/calibration.py`**: Contains functions for:
    * Capturing calibration images from a specified camera.
    * Performing camera calibration using `cv2.calibrateCamera()`.
    * Saving and loading camera-specific calibration parameters (e.g., `camera_params_idx{index}.npz`).
* **`src/feature_utils.py`**: Provides utility functions for:
    * Detecting features (e.g., ORB).
    * Matching features between two sets of descriptors.
* **`src/sfm.py`**: Implements Structure from Motion functionalities:
    * `estimate_pose()`: Estimates relative camera pose (R, t).
    * `triangulate_points()`: Reconstructs 3D points.

## Configuration Constants

Several parameters can be adjusted by modifying the constants defined in the respective Python files:

* **`src/calibration.py`**:
    * `CHECKERBOARD_SIZE`: Dimensions of the internal corners of your checkerboard.
    * `SQUARE_SIZE_MM`: The side length of a square on your checkerboard in millimeters.
    * `MAX_IMAGES`: Maximum number of calibration images to capture.
    * `AUTO_CAPTURE_INTERVAL_SECONDS`: Time interval between automatic image captures during calibration.
    * `MIN_CALIBRATION_IMAGES`: Minimum number of calibration images required for calibration.
    * `CALIBRATION_IMAGE_DIR`: Directory where calibration images and parameter files are stored.
* **`src/main.py`**:
    * `MIN_MATCHES_FOR_POSE`: Minimum number of good feature matches required to attempt pose estimation.
    * The calibration file path is no longer a single static `CALIBRATION_FILE` constant but is dynamically generated based on the selected camera index (e.g., `calibration_data/camera_params_idx{selected_camera_index}.npz`).
* **`src/feature_utils.py`**:
    * `ratio_thresh`: Lowe's ratio test threshold for good matches.

This project provides a foundation for experimenting with visual SLAM techniques.
