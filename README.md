# SLAM Experiment Project

This project implements a basic Simultaneous Localization and Mapping (SLAM) pipeline using Python and OpenCV. It includes modules for camera calibration, feature detection and matching, and 2-view structure from motion (SfM) to estimate camera pose and triangulate 3D points.

## Project Structure

```
slam-experiment/
├── requirements.txt
├── src/
│   ├── calibration.py    # Handles camera calibration
│   ├── feature_utils.py  # Utilities for feature detection and matching
│   ├── main.py           # Main application script
│   └── sfm.py            # Functions for Structure from Motion (pose estimation, triangulation)
└── calibration_data/     # Directory to save calibration images and parameters (created automatically)
```

## Features

* **Camera Calibration**: Captures checkerboard images and calculates camera intrinsic matrix and distortion coefficients.
* **Feature Detection**: Uses ORB detector to find keypoints in images.
* **Feature Matching**: Matches features between consecutive frames using a brute-force matcher with ratio test.
* **Pose Estimation**: Estimates camera rotation and translation between two views using the 5-point algorithm for the Essential Matrix and RANSAC.
* **3D Triangulation**: Reconstructs 3D points from matched 2D keypoints and estimated poses.
* **Live Feed**: Shows live webcam feed with undistorted images, detected keypoints, and feature matches.

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
The primary dependencies are `opencv-python` and `numpy`.

## Running the Application

To run the SLAM experiment, execute the `main.py` script from the root directory of the project:

```bash
python src/main.py
```

### Workflow:

1.  **Calibration Data Check**: The application first attempts to load existing camera calibration data from `calibration_data/camera_params.npz`.
2.  **Camera Calibration (if needed)**:
    * If calibration data is not found or fails to load, the camera calibration process will start.
    * A live feed from your webcam will be displayed.
    * Position a checkerboard (default internal corners: 12x8) in front of the camera.
    * Press **SPACE** to capture an image when checkerboard corners are detected and clearly visible. The system requires a minimum of 5 images (configurable by `MIN_CALIBRATION_IMAGES`) and aims to capture up to 15 images (configurable by `MAX_IMAGES`).
    * Captured images are saved in the `calibration_data/` directory.
    * Press **'q'** to finish capturing images and proceed to calibration (if enough images are captured) or to quit the capture mode.
    * Press **ESC** to discard all captures and exit the calibration image capture process.
    * If calibration is successful, the parameters (`camera_matrix`, `dist_coeffs`, `frame_size`, `reprojection_error`) are saved to `calibration_data/camera_params.npz`.
3.  **Feature Detection and Matching Loop**:
    * Once the camera is calibrated (or existing data is loaded), the application starts a live video feed for feature tracking.
    * The frames are undistorted using the calibration parameters.
    * ORB features are detected in each frame.
    * Features are matched between the current and previous frames.
    * Matches are displayed in a separate window titled "Feature Matches".
    * Keypoints are drawn on the live undistorted feed in a window titled "Live Feed with Keypoints".
4.  **Two-View Reconstruction**:
    * If a sufficient number of good matches (default: `MIN_MATCHES_FOR_POSE = 10`) are found, the application attempts to:
        * Estimate the relative camera pose (rotation and translation) between the previous and current frames.
        * Triangulate 3D points from the matched 2D features using the estimated pose.
    * Information about pose estimation and triangulation success/failure and the number of reconstructed points will be printed to the console.
5.  **Exiting**:
    * Press **'q'** in the live feed window to quit the feature tracking loop and end the application.

## Key Files and Modules

* **`src/main.py`**: The main entry point that orchestrates the calibration and feature tracking/SfM pipeline.
* **`src/calibration.py`**: Contains functions for:
    * Capturing calibration images from a webcam.
    * Performing camera calibration using `cv2.calibrateCamera()`.
    * Saving and loading calibration parameters (`.npz` file).
* **`src/feature_utils.py`**: Provides utility functions for:
    * Detecting features (e.g., ORB) using `cv2.ORB_create().detectAndCompute()`.
    * Matching features between two sets of descriptors using `cv2.BFMatcher()` and Lowe's ratio test.
* **`src/sfm.py`**: Implements Structure from Motion functionalities:
    * `estimate_pose()`: Estimates camera pose (R, t) from 2D point correspondences using `cv2.findEssentialMat()` and `cv2.recoverPose()`.
    * `triangulate_points()`: Reconstructs 3D points from 2D correspondences and projection matrices using `cv2.triangulatePoints()`.

## Configuration Constants

Several parameters can be adjusted by modifying the constants defined in the respective Python files:

* **`src/calibration.py`**:
    * `CHECKERBOARD_SIZE`: Dimensions of the internal corners of your checkerboard (e.g., `(12, 8)`).
    * `SQUARE_SIZE_MM`: The side length of a square on your checkerboard in millimeters.
    * `MAX_IMAGES`: Maximum number of calibration images to capture.
    * `MIN_CALIBRATION_IMAGES`: Minimum number of calibration images required for calibration.
* **`src/main.py`**:
    * `CALIBRATION_FILE`: Path to the saved calibration parameters file.
    * `MIN_MATCHES_FOR_POSE`: Minimum number of good feature matches required to attempt pose estimation.
* **`src/feature_utils.py`**:
    * `ratio_thresh`: Lowe's ratio test threshold for good matches (default is 0.75).
