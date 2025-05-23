"""
Performs camera calibration using checkerboard images.

This script can be run directly to calibrate a camera using images in a specified
directory. Its core calibration logic is also designed to be reusable by other
scripts, such as the calibration_assistant.py.

When run directly, it uses predefined constants for checkerboard dimensions,
square size, image directory, and output file.
"""
import cv2
import numpy as np
import os
import yaml
from pathlib import Path # Using pathlib for path operations

# --- Default Configuration Constants (used when script is run directly) ---
DEFAULT_CHECKERBOARD_INTERNAL_CORNERS = (12, 8) # (cols-1, rows-1) or (width-1, height-1)
DEFAULT_SQUARE_SIZE_MM = 20.0 # Physical size of a square in millimeters
DEFAULT_CALIBRATION_IMAGE_DIR = "data/calibration_images/"
DEFAULT_CALIBRATION_DATA_FILE = "data/camera_calibration.yaml"

# --- User Instructions ---
USER_INSTRUCTIONS_TEMPLATE = """
Camera Calibration Script
-------------------------

This script calibrates a camera using checkerboard images.

1.  Prepare a checkerboard pattern.
    - The script expects inner corner dimensions (width-1, height-1).
    - The physical size of each square on the checkerboard is also needed.

    When run directly, defaults are:
    - Checkerboard Internal Corners: {} (Width-1, Height-1)
    - Square Size: {} mm

2.  Capture images of the checkerboard:
    * Use a good quality camera.
    * Capture about 15-20 images.
    * Show the checkerboard from various angles and distances.
    * Ensure the entire checkerboard is visible.
    * Keep the checkerboard flat and well-lit. Avoid glare.

3.  Place captured images (e.g., .png, .jpg) into an image directory.
    Default directory when run directly: `{}`

4.  Run this script (or `calibration_assistant.py` for interactive capture).
    The calibration results will be saved to a YAML file.
    Default output file when run directly: `{}`
"""

def perform_calibration_from_images(
    images_dir_path: str,
    checkerboard_dims: tuple[int, int],
    square_size_mm: float,
    output_yaml_filepath: str
) -> dict | None:
    """
    Performs camera calibration using collected images and saves the results.

    Args:
        images_dir_path (str): Directory containing calibration images.
        checkerboard_dims (tuple[int, int]): (Width, Height) of inner checkerboard corners.
        square_size_mm (float): The size of a checkerboard square in millimeters.
        output_yaml_filepath (str): Path to save the calibration results YAML file.

    Returns:
        dict | None: A dictionary with calibration results ('camera_matrix', 'dist_coeffs',
                      'reprojection_error', 'image_width', 'image_height', 'num_valid_images',
                      'checkerboard_dims', 'square_size_mm')
                      or None if calibration fails.
    """
    print(f"\nStarting calibration process...")
    print(f"  Image directory: {images_dir_path}")
    print(f"  Checkerboard inner corners: {checkerboard_dims}")
    print(f"  Square size: {square_size_mm} mm")
    print(f"  Output file: {output_yaml_filepath}")

    # Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), ..., (width-1,height-1,0)
    # These are 3D points in the checkerboard's own coordinate system.
    objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
    objp = objp * square_size_mm # Scale by the actual square size

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    image_dir = Path(images_dir_path)
    if not image_dir.exists():
        print(f"Error: Calibration image directory not found: {image_dir_path}")
        return None

    image_files = list(image_dir.glob("*.png")) + \
                  list(image_dir.glob("*.jpg")) + \
                  list(image_dir.glob("*.jpeg"))

    if not image_files:
        print(f"No .png, .jpg, or .jpeg images found in '{images_dir_path}'. Cannot calibrate.")
        return None

    print(f"\nFound {len(image_files)} images for calibration processing in '{images_dir_path}'.")

    img_width, img_height = 0, 0
    valid_images_processed = 0

    for i, fname_path in enumerate(image_files):
        img = cv2.imread(str(fname_path))
        print(f"Processing image {i+1}/{len(image_files)}: {fname_path.name}...", end=" ")

        if img is None:
            print("Failed to load. Skipping.")
            continue

        current_h, current_w = img.shape[:2]
        if valid_images_processed == 0:
            img_width, img_height = current_w, current_h
            print(f"Dimensions set to {img_width}x{img_height}.", end=" ")
        elif (img_width != current_w) or (img_height != current_h):
            print(f"Mismatched dimensions ({current_w}x{current_h} vs expected {img_width}x{img_height}). Skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)

        if ret:
            print("Checkerboard found.", end=" ")
            objpoints.append(objp)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_subpix)
            valid_images_processed += 1
            # Optional: Draw and display corners
            # cv2.drawChessboardCorners(img, checkerboard_dims, corners_subpix, ret)
            # cv2.imshow(f'Corners in {fname_path.name}', cv2.resize(img, (img_width//2, img_height//2)))
            # cv2.waitKey(100)
        else:
            print("Checkerboard NOT found.")
    # cv2.destroyAllWindows()

    if valid_images_processed == 0:
        print("\nNo checkerboards found in any of the processed images. Calibration cannot proceed.")
        return None
    
    min_required_images = 5 # A more practical minimum for cv2.calibrateCamera
    if valid_images_processed < min_required_images:
        print(f"\nWarning: Checkerboards successfully processed in only {valid_images_processed} images.")
        print(f"At least {min_required_images} images with detected checkerboards are recommended for a stable calibration.")
        if input("Continue with calibration anyway? (y/n): ").lower() != 'y':
            print("Calibration aborted by user due to too few valid images.")
            return None

    print(f"\nCalibrating camera using {valid_images_processed} valid images (image size: {img_width}x{img_height})...")
    try:
        ret_cal, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img_width, img_height), None, None)

        if not ret_cal:
            print("cv2.calibrateCamera returned False. Calibration failed.")
            return None

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        reprojection_error = mean_error / len(objpoints)

        calibration_data = {
            "camera_matrix": mtx, # Keep as numpy array for now
            "dist_coeffs": dist,  # Keep as numpy array for now
            "reprojection_error": reprojection_error,
            "image_width": img_width,
            "image_height": img_height,
            "num_valid_images": valid_images_processed,
            "checkerboard_dims": checkerboard_dims, # Save the dimensions used
            "square_size_mm": square_size_mm # Save the square size used
        }

        # Save to YAML
        save_path_obj = Path(output_yaml_filepath)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for YAML (convert numpy arrays to lists)
        data_to_save_yaml = {
            "image_width": calibration_data["image_width"],
            "image_height": calibration_data["image_height"],
            "camera_matrix": calibration_data["camera_matrix"].tolist(),
            "dist_coeffs": calibration_data["dist_coeffs"].tolist(),
            "reprojection_error": calibration_data["reprojection_error"],
            "num_valid_images_for_calibration": calibration_data["num_valid_images"],
            "square_size_mm": calibration_data["square_size_mm"]
        }

        with open(save_path_obj, 'w') as f:
            yaml.dump(data_to_save_yaml, f, sort_keys=False, default_flow_style=None)
        print(f"\nCalibration results successfully saved to '{output_yaml_filepath}'.")
        print(f"  Camera Matrix (K):\n{calibration_data['camera_matrix']}")
        print(f"  Distortion Coefficients:\n{calibration_data['dist_coeffs']}")
        print(f"  Reprojection Error: {calibration_data['reprojection_error']:.4f}")

        return calibration_data # Return the data with numpy arrays for potential direct use

    except cv2.error as e:
        print(f"OpenCV Error during cv2.calibrateCamera: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during calibration: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main function called when the script is run directly.
    Uses default constants for calibration.
    """
    # Format instructions with current default values
    formatted_instructions = USER_INSTRUCTIONS_TEMPLATE.format(
        DEFAULT_CHECKERBOARD_INTERNAL_CORNERS,
        DEFAULT_CHECKERBOARD_INTERNAL_CORNERS[0] + 1, DEFAULT_CHECKERBOARD_INTERNAL_CORNERS[1] + 1,
        DEFAULT_SQUARE_SIZE_MM,
        os.path.abspath(DEFAULT_CALIBRATION_IMAGE_DIR),
        os.path.abspath(DEFAULT_CALIBRATION_DATA_FILE)
    )
    print(formatted_instructions)

    # Ensure the default image directory exists
    Path(DEFAULT_CALIBRATION_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Prompt user to confirm before proceeding
    if input(f"Proceed with calibration using images from '{DEFAULT_CALIBRATION_IMAGE_DIR}' and default settings? (y/n): ").lower() != 'y':
        print("Calibration aborted by user.")
        return

    perform_calibration_from_images(
        images_dir_path=DEFAULT_CALIBRATION_IMAGE_DIR,
        checkerboard_dims=DEFAULT_CHECKERBOARD_INTERNAL_CORNERS,
        square_size_mm=DEFAULT_SQUARE_SIZE_MM,
        output_yaml_filepath=DEFAULT_CALIBRATION_DATA_FILE
    )

if __name__ == '__main__':
    main()
