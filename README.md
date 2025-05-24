# Real-Time Monocular 3D Reconstruction

This project implements a real-time 3D reconstruction system using a monocular camera. It tracks features across video frames, estimates camera motion, and triangulates 3D points to build a sparse point cloud representation of the observed scene. The system is built incrementally, starting with a two-view Structure from Motion (SfM) initialization and then extending to new frames.

## Directory Structure

*   `src/`: Contains the core source code for the SfM system.
    *   `sfm_system.py`: The main module orchestrating the incremental SfM process, including feature tracking, pose estimation (PnP), and 3D point triangulation.
*   `scripts/`: Contains utility scripts for running different parts of the application.
    *   `calibrate_camera.py`: Script for performing camera calibration.
    *   `main.py`: The main entry point to run the full 3D reconstruction system.
*   `calibration_params.npz`: (Generated File) Stores the camera matrix and distortion coefficients after running `calibrate_camera.py`. This file is crucial for the SfM system.

## Setup and Usage

### Dependencies

The project requires the following Python libraries:
*   OpenCV (`cv2`)
*   NumPy (`numpy`)
*   Matplotlib

You can typically install these using pip:
```bash
pip install opencv-python numpy matplotlib
```
Ensure you also have a compatible version of Python installed (e.g., Python 3.7+).

### Camera Calibration

Accurate camera calibration is essential for good 3D reconstruction results.

1.  **Prepare a Checkerboard:** You will need a checkerboard pattern with **12x8 inner corners**. This means the checkerboard should have 13x9 squares.
2.  **Run the Calibration Script:**
    ```bash
    python scripts/calibrate_camera.py
    ```
3.  **Follow On-Screen Instructions:**
    *   The script will open a camera feed.
    *   Position the 12x8 checkerboard in front of the camera.
    *   When the corners are clearly detected (drawn on screen), press the 'c' key to capture the image.
    *   Capture at least 10-15 images from different angles and distances to ensure a good calibration.
    *   Once enough images are captured, press 'q' to quit the capture mode and perform the calibration.
4.  **Output:** The script will save the camera matrix and distortion coefficients to a file named `calibration_params.npz` in the root directory of the project. This file will be automatically loaded by the main SfM system.

### Running the Application

After successful camera calibration (and `calibration_params.npz` is present):

1.  **Run the Main Script:**
    ```bash
    python scripts/main.py
    ```
2.  **System Operation:**
    *   The script will first prompt you to confirm that the existing `calibration_params.npz` is appropriate for your current camera setup.
    *   It will then initialize the SfM system, typically requiring you to capture two initial frames by pressing 'c' to establish the initial 3D map.
    *   After initialization, the system will continuously process new frames, track features, estimate camera pose, and update a 3D point cloud visualization in real-time.
    *   Press 'q' in the OpenCV window to stop the application.

## Core Components

*   **`src/sfm_system.py`**: This is the heart of the 3D reconstruction pipeline. It handles:
    *   Initialization of the 3D map from two views.
    *   Tracking features in new frames.
    *   Estimating camera pose for new frames using Perspective-n-Point (PnP) against the existing 3D map.
    *   Triangulating new 3D points by matching features between frames with known poses.
    *   Managing the 3D world points and camera poses.
    *   Real-time 3D visualization of the point cloud and camera trajectory using Matplotlib.
    *   (Placeholder for Bundle Adjustment for global optimization).

*   **`scripts/calibrate_camera.py`**: A utility to calculate the intrinsic parameters (camera matrix and distortion coefficients) of your specific camera. It uses a checkerboard pattern and saves the results to `calibration_params.npz`. The dimensions of the checkerboard (number of inner corners) are currently set to 12x8 in this script.

*   **`scripts/main.py`**: The entry script that ensures calibration data is available and then launches the `sfm_system`.

The other Python files in the root directory (`camera_feed.py`, `feature_tracker.py`, `two_view_sfm.py`) represent earlier, standalone development stages of the components that were later integrated into `sfm_system.py`.
