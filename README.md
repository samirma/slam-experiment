# SLAM Experiment Project

This project implements a Simultaneous Localization and Mapping (SLAM) system using Python.
It now utilizes the `pyslam` library (https://github.com/luigifreda/pyslam) for its core SLAM functionality, replacing the previous custom implementation.
The system supports multiple cameras and performs per-camera calibration.

## Project Structure

```
slam-experiment/
├── config/
│   ├── pyslam_config.yaml                    # Main configuration for pyslam
│   └── pyslam_settings_idx{CAM_IDX}.yaml     # Camera-specific settings for pyslam (generated)
├── requirements.txt
├── README.md
├── src/
│   ├── calibration.py    # Handles camera calibration & generation of pyslam camera settings
│   ├── main.py           # Main application script (integrates calibration and pyslam)
│   └── camera_selection.py # Utility for selecting camera
└── calibration_data/     # Directory to save raw calibration images and original parameters (e.g., camera_params_idx0.npz)
```

## Core Changes & Features

* **`pyslam` Integration**: This project now utilizes the `pyslam` library for its core SLAM functionality.
* **Multi-Camera Support**: Users can select from a list of available cameras connected to the system.
* **Per-Camera Calibration**: Captures checkerboard images and calculates camera intrinsic matrix and distortion coefficients for the selected camera.
    * Standard calibration data is saved (e.g., `calibration_data/camera_params_idx0.npz`).
    * Camera calibration parameters for `pyslam` are also stored in `config/pyslam_settings_idx{CAMERA_INDEX}.yaml` (where `{CAMERA_INDEX}` is the selected camera index), generated automatically after camera calibration.
* **Configuration**:
    * The behavior of `pyslam` itself is configured via `config/pyslam_config.yaml`. You can edit this file to change SLAM parameters, feature types, enable/disable the viewer, etc., according to `pyslam`'s documentation.
* **Live Feed**: Shows the live undistorted webcam feed that is being provided to `pyslam`.
* **3D Visualization**: The previous 3D visualization based on Open3D has been replaced by `pyslam`'s own visualization system (typically using Pangolin). This should activate if `kUseViewer: True` is set in `config/pyslam_config.yaml`.

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
The primary dependencies are `numpy`, `opencv-python` (for core computer vision tasks and calibration), `PyYAML` (for handling `pyslam` configurations), and `pyslam` itself (which you'll install from its repository).

## Additional Dependencies (`pyslam`)

As mentioned, this project now relies on `pyslam` for its core SLAM operations. `pyslam` is a separate, comprehensive SLAM library that needs to be installed from its own repository.

**Key points regarding `pyslam` integration:**
*   This project utilizes the `pyslam` library (https://github.com/luigifreda/pyslam) for its core SLAM functionality, replacing the previous custom implementation.
*   Camera calibration parameters for `pyslam` are stored in `config/pyslam_settings_idx{CAMERA_INDEX}.yaml` (where `{CAMERA_INDEX}` is the selected camera index), generated automatically after camera calibration.
*   The behavior of `pyslam` itself is configured via `config/pyslam_config.yaml`. You can edit this file to change SLAM parameters, feature types, enable/disable the viewer, etc., according to `pyslam`'s documentation.
*   The previous 3D visualization based on Open3D has been replaced by `pyslam`'s own visualization system (typically using Pangolin), which should activate if `kUseViewer: True` is set in `config/pyslam_config.yaml`.

To install `pyslam`:

1.  **Clone the `pyslam` repository recursively:**
    ```bash
    git clone --recursive https://github.com/luigifreda/pyslam.git
    ```
2.  **Navigate into the cloned `pyslam` directory:**
    ```bash
    cd pyslam
    ```
3.  **Run the installation script:**
    ```bash
    ./install_all.sh
    ```
    This script might create a Python virtual environment (e.g., `pyslam_env`) or use Conda. Please follow the instructions and output provided by the `pyslam` installation script carefully.

4.  **Activate `pyslam` environment (if necessary):**
    If `pyslam` was installed into its own virtual environment or Conda environment, you will need to activate it before running the main application if you intend to use `pyslam`-dependent features.

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
    *   If calibration is successful, the parameters (`camera_matrix`, `dist_coeffs`, `frame_size`, `reprojection_error`) are saved to a camera-specific file in the `calibration_data/` directory (e.g., `camera_params_idx0.npz`). Additionally, a `pyslam_settings_idx{CAMERA_INDEX}.yaml` file is generated in the `config/` directory for use by `pyslam`.
4.  **SLAM Processing with `pyslam`**:
    *   Once the selected camera is calibrated and its `pyslam` settings file is generated, the application initializes `pyslam`.
    *   **Live Feed Window**: An OpenCV window ("Live Feed to pyslam") shows the undistorted live video from the selected camera that is being fed to `pyslam`.
    *   **`pyslam` Viewer**: If `kUseViewer: True` is set in `config/pyslam_config.yaml`, `pyslam`'s own visualization window (typically Pangolin-based) should open and display the map, camera trajectory, and features.
    *   **Console Output**: Information from `pyslam` (if any) and status messages from `main.py` will be printed to the console.
5.  **Exiting**:
    *   Press **'q'** in the "Live Feed to pyslam" OpenCV window to quit the main loop.
    *   Closing `pyslam`'s viewer window may also terminate the application, depending on `pyslam`'s behavior.

## Key Files and Modules

* **`src/main.py`**: The main entry point. Handles camera selection, orchestrates the per-camera calibration process, initializes and manages the `pyslam` system.
* **`src/calibration.py`**: Contains functions for:
    * Capturing calibration images from a specified camera.
    * Performing camera calibration using `cv2.calibrateCamera()`.
    * Saving camera-specific calibration parameters (both the original `.npz` and the `pyslam`-specific `.yaml` in `config/`).
* **`src/camera_selection.py`**: Utility for detecting and selecting cameras.
* **`config/pyslam_config.yaml`**: Main configuration file for `pyslam`. Edit this to control SLAM parameters, feature types, viewer, etc.
* **`config/pyslam_settings_idx{CAMERA_INDEX}.yaml`**: Camera-specific calibration files for `pyslam`, generated automatically.

## Configuration

* **Camera Calibration Parameters (`src/calibration.py`)**:
    * `CHECKERBOARD_SIZE`: Dimensions of the internal corners of your checkerboard.
    * `SQUARE_SIZE_MM`: The side length of a square on your checkerboard in millimeters.
    * `MAX_IMAGES`: Maximum number of calibration images to capture.
    * `AUTO_CAPTURE_INTERVAL_SECONDS`: Time interval between automatic image captures during calibration.
    * `MIN_CALIBRATION_IMAGES`: Minimum number of calibration images required for calibration.
    * `CALIBRATION_IMAGE_DIR`: Directory where raw calibration images and `.npz` parameter files are stored.
* **`pyslam` SLAM Parameters (`config/pyslam_config.yaml`)**:
    * This file is passed to `pyslam`. Refer to `pyslam`'s documentation for details on its various configuration options (e.g., `kFeatureType`, `kUseLoopClosing`, `kUseViewer`, ORB extractor settings, etc.).
    * The `DATASET.FOLDER_DATASET.settings_file` path within this config is updated dynamically by `main.py` to point to the correct camera-specific YAML file (e.g., `config/pyslam_settings_idx0.yaml`).

This project provides a foundation for experimenting with `pyslam` using a live camera feed.

## Troubleshooting

If you encounter issues, consider the following:

*   **`pyslam` Installation:** Ensure `pyslam` and all its dependencies (including Pangolin for the viewer) are correctly installed. Follow the installation instructions in the `pyslam` repository carefully.
*   **Camera Access:** Verify your camera is accessible by OpenCV. Errors like "Could not open webcam" usually point to camera connection or permission issues.
*   **`pyslam` Configuration:** Check `config/pyslam_config.yaml` and the generated `config/pyslam_settings_idx{CAMERA_INDEX}.yaml`. Incorrect paths or parameters can cause `pyslam` to fail.
*   **`pyslam` Viewer Issues:** If `pyslam`'s viewer does not appear or crashes, this is likely an issue within `pyslam` or its Pangolin dependency (graphics drivers, Wayland/X11 compatibility, etc.). Consult `pyslam`'s documentation or issue tracker.
