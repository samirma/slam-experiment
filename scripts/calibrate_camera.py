"""
Performs camera calibration using checkerboard images.

This script guides the user through capturing checkerboard images,
processes these images to find checkerboard corners, and then uses
OpenCV's camera calibration functions to estimate the camera matrix (K)
and distortion coefficients.

The calibration results are saved to a YAML file (`data/camera_calibration.yaml`),
which can then be loaded by the `CameraParams` class in `src/utils/camera_params.py`
for use in other parts of the SLAM system.
"""
import cv2
import numpy as np
import os
import glob
import yaml # Using YAML for saving data

# --- Chessboard Parameters ---
# Redefine CHECKERBOARD to be (cols, rows) to match cv2.findChessboardCorners behavior
# It expects (patternSize.width, patternSize.height)
# Number of internal corners:
# e.g., for a 7x10 board (7 squares wide, 10 squares high), it has 6x9 internal corners.
CHECKERBOARD_INTERNAL_CORNERS = (12, 8) # (cols-1, rows-1) or (width-1, height-1)
SQUARE_SIZE_MM = 20.0 # Physical size of a square in millimeters

# --- Calibration Image Path ---
# Users should place their calibration images in this directory.
CALIBRATION_IMAGE_DIR = "data/calibration_images/"
CALIBRATION_DATA_FILE = "data/camera_calibration.yaml"

# --- User Instructions ---
USER_INSTRUCTIONS = """
Camera Calibration Script
-------------------------

1.  Prepare a checkerboard pattern with known dimensions.
    The script is currently configured for a checkerboard with {} internal corners
    (e.g., a board with {}x{} squares).
    The size of each square is set to {} mm.
    Adjust `CHECKERBOARD_INTERNAL_CORNERS` and `SQUARE_SIZE_MM` in the script if needed.

    Default internal corners: {} (Width-1, Height-1)
    Default square size: {} mm

2.  Capture images of the checkerboard:
    *   Use a good quality camera.
    *   Capture about 15-20 images.
    *   Show the checkerboard from various angles and distances.
    *   Ensure the entire checkerboard is visible in the images.
    *   Keep the checkerboard flat.
    *   Good lighting is important. Avoid glare.

3.  Place the captured images (e.g., .png, .jpg) into the directory:
    `{}`

4.  Run this script. It will process the images, perform calibration,
    and save the results to:
    `{}`
""".format(CHECKERBOARD_INTERNAL_CORNERS, # This will be the new (12,8)
           CHECKERBOARD_INTERNAL_CORNERS[0] + 1, CHECKERBOARD_INTERNAL_CORNERS[1] + 1, # This becomes 13x9
           SQUARE_SIZE_MM,
           CHECKERBOARD_INTERNAL_CORNERS, # Explicitly state the default in the instructions
           SQUARE_SIZE_MM,                # Explicitly state the default in the instructions
           os.path.abspath(CALIBRATION_IMAGE_DIR),
           os.path.abspath(CALIBRATION_DATA_FILE))


def calibrate_camera():
    print(USER_INSTRUCTIONS)

    # --- Prepare Object Points ---
    # These are the 3D coordinates of the chessboard corners in its own coordinate system (Z=0).
    objp = np.zeros((CHECKERBOARD_INTERNAL_CORNERS[0] * CHECKERBOARD_INTERNAL_CORNERS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_INTERNAL_CORNERS[0], 0:CHECKERBOARD_INTERNAL_CORNERS[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

    # Arrays to store object points and image points from all images.
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # --- Image Acquisition ---
    if not os.path.exists(CALIBRATION_IMAGE_DIR):
        print(f"Error: Calibration image directory not found: {CALIBRATION_IMAGE_DIR}")
        print("Please create it and add calibration images.")
        os.makedirs(CALIBRATION_IMAGE_DIR, exist_ok=True) # Create if not exist
        return

    image_paths = glob.glob(os.path.join(CALIBRATION_IMAGE_DIR, '*.png'))
    image_paths.extend(glob.glob(os.path.join(CALIBRATION_IMAGE_DIR, '*.jpg')))
    image_paths.extend(glob.glob(os.path.join(CALIBRATION_IMAGE_DIR, '*.jpeg')))


    if not image_paths:
        print(f"No images found in {CALIBRATION_IMAGE_DIR}. Please add calibration images.")
        return

    print(f"\nFound {len(image_paths)} images for calibration.")
    processed_images = 0
    successful_detections = 0

    # --- Corner Detection and Calibration ---
    for fname in image_paths:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}. Skipping.")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_images += 1

        # Find the chessboard corners
        # Flags like cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE can sometimes help.
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_INTERNAL_CORNERS, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            successful_detections += 1
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners (optional)
            cv2.drawChessboardCorners(img, CHECKERBOARD_INTERNAL_CORNERS, corners2, ret)
            # cv2.imshow(f'Chessboard Detections - {os.path.basename(fname)}', img)
            # cv2.waitKey(500) # Display for 0.5 seconds
            print(f"Successfully found corners in {os.path.basename(fname)}")
        else:
            print(f"Could not find corners in {os.path.basename(fname)}")

    # cv2.destroyAllWindows()

    if successful_detections < 10: # Need at least a few good views
        print(f"\nCalibration failed: Only {successful_detections} images had successful corner detections.")
        print("Need at least 10 good images with diverse views of the checkerboard.")
        return

    print(f"\nPerforming calibration with {successful_detections} successfully processed images...")

    # --- Perform Calibration ---
    # gray.shape[::-1] gives (width, height)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        print("Calibration failed! cv2.calibrateCamera returned False.")
        return

    print("\n--- Calibration Results ---")
    print("Camera Matrix (K):")
    print(mtx)
    print("\nDistortion Coefficients (k1, k2, p1, p2, k3):")
    print(dist)

    # --- Calculate Re-projection Error ---
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    reprojection_error = mean_error / len(objpoints)
    print(f"\nMean Re-projection Error: {reprojection_error}")
    if reprojection_error > 1.0:
        print("Warning: Re-projection error is high. Calibration might not be accurate.")
        print("Consider retaking images, ensuring checkerboard flatness, diverse views, and correct parameters.")
    else:
        print("Re-projection error is acceptable.")


    # --- Save Calibration Data ---
    calibration_data = {
        'camera_matrix': mtx.tolist(), # Convert numpy array to list for YAML serialization
        'dist_coeffs': dist.tolist(),
        'image_width': gray.shape[1], # width
        'image_height': gray.shape[0], # height
        'checkerboard_internal_corners': CHECKERBOARD_INTERNAL_CORNERS,
        'square_size_mm': SQUARE_SIZE_MM,
        'reprojection_error': reprojection_error
    }

    # Ensure data directory exists
    os.makedirs(os.path.dirname(CALIBRATION_DATA_FILE), exist_ok=True)

    try:
        with open(CALIBRATION_DATA_FILE, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False)
        print(f"\nCalibration data saved successfully to: {os.path.abspath(CALIBRATION_DATA_FILE)}")
    except Exception as e:
        print(f"\nError saving calibration data: {e}")

if __name__ == '__main__':
    calibrate_camera()
