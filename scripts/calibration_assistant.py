import cv2
import os
import numpy as np
import yaml
import glob
from pathlib import Path
import argparse

# Constants
CALIBRATION_IMAGES_DIR = "data/calibration_images"
CALIBRATION_RESULTS_FILE = "data/camera_calibration.yaml"
DEFAULT_CHECKERBOARD_SIZE = (9, 6) # Common inner corners (width, height)
MIN_IMAGES_FOR_CALIBRATION = 10

def list_available_cameras(max_cameras_to_check=10):
    """
    Detects available camera devices that OpenCV can access.

    Args:
        max_cameras_to_check (int): Maximum number of camera indices to try.

    Returns:
        list[int]: A list of available camera IDs.
    """
    available_cameras = []
    print("Detecting available cameras...")
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
            print(f"  Camera ID {i} is available.")
        else:
            # Some systems might "reserve" an ID even if not fully available,
            # or might be slow to release. Breaking early if a non-contiguous ID fails.
            # However, some systems might have 0, 2 available but not 1.
            # For now, we'll continue checking up to max_cameras_to_check.
            print(f"  Camera ID {i} not accessible or does not exist.")
    
    if not available_cameras:
        print("No cameras found.")
    else:
        print(f"Found camera IDs: {available_cameras}")
    return available_cameras

def select_camera_id(available_cameras):
    """
    Prompts the user to select a camera ID from the list of available cameras.

    Args:
        available_cameras (list[int]): List of detected camera IDs.

    Returns:
        int | None: The selected camera ID, or None if no selection is made.
    """
    if not available_cameras:
        return None

    while True:
        try:
            if len(available_cameras) == 1:
                print(f"Automatically selecting camera ID {available_cameras[0]}.")
                return available_cameras[0]
            
            user_input = input(f"Please enter the camera ID to use for calibration from {available_cameras}: ")
            selected_id = int(user_input)
            if selected_id in available_cameras:
                print(f"Camera ID {selected_id} selected.")
                return selected_id
            else:
                print(f"Invalid ID. Please choose from {available_cameras}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

def main_cli(cli_checkerboard_dims_override: tuple[int, int] | None = None):
    """
    Main command-line interface function for the calibration assistant.

    Args:
        cli_checkerboard_dims_override (tuple[int, int] | None): 
            If provided via CLI, these dimensions will be used, bypassing the prompt.
    """
    print("Welcome to the Camera Calibration Assistant!")

    # 1. List Available Cameras and Select
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("Exiting: No cameras available for calibration.")
        return

    camera_id = select_camera_id(available_cameras)
    if camera_id is None:
        print("Exiting: No camera selected.")
        return

    print(f"\nProceeding with Camera ID: {camera_id}")

    # 2. Image Collection
    print_image_collection_instructions()
    
    # Prepare directory for images
    prepare_calibration_image_dir(CALIBRATION_IMAGES_DIR) # Clear or create
    
    num_images_collected = collect_calibration_images(camera_id, CALIBRATION_IMAGES_DIR)

    if num_images_collected == 0: 
        print("\nNo images were collected. Cannot proceed to calibration.")
        return
        
    if num_images_collected < MIN_IMAGES_FOR_CALIBRATION:
        print(f"\nWarning: Only {num_images_collected} images collected, which is less than the recommended {MIN_IMAGES_FOR_CALIBRATION}.")
        print("Calibration quality may be poor.")
        if input("Do you want to continue with calibration anyway? (y/n): ").lower() != 'y':
            print("Calibration aborted by user.")
            return
    
    # Get checkerboard size
    if cli_checkerboard_dims_override:
        checkerboard_dims = cli_checkerboard_dims_override
        print(f"\nUsing checkerboard dimensions from command line: {checkerboard_dims}")
    else:
        # Prompt user for checkerboard dimensions if not provided via CLI
        checkerboard_dims = get_checkerboard_dimensions(DEFAULT_CHECKERBOARD_SIZE)
        print(f"Using checkerboard dimensions: {checkerboard_dims}")


    print(f"\nRunning calibration with checkerboard size: {checkerboard_dims}...")
    calibration_data = run_calibration(CALIBRATION_IMAGES_DIR, checkerboard_dims)

    if calibration_data:
        print("\nCalibration successful!")
        print(f"  Camera Matrix (K):\n{calibration_data['camera_matrix']}")
        print(f"  Distortion Coefficients:\n{calibration_data['dist_coeffs']}")
        print(f"  Reprojection Error: {calibration_data['reprojection_error']:.4f}")
        print(f"  Image Dimensions: {calibration_data['image_width']}x{calibration_data['image_height']}")
        print(f"  Calibration performed using {calibration_data['num_valid_images']} valid images.")
        
        # 4. Display and Save Results
        save_calibration_results(CALIBRATION_RESULTS_FILE, calibration_data)

        if input("\nDo you want to see an undistortion preview using one of the calibration images? (y/n): ").lower() == 'y':
            display_undistortion_preview(
                calibration_data['camera_matrix'],
                calibration_data['dist_coeffs'],
                CALIBRATION_IMAGES_DIR,
                (calibration_data['image_width'], calibration_data['image_height'])
            )
        print(f"\nCalibration complete. Results saved to '{CALIBRATION_RESULTS_FILE}'.")

    else:
        print("\nCalibration failed. Please check the collected images and checkerboard pattern.")
        return


def save_calibration_results(filepath: str, calib_data: dict):
    """
    Saves the calibration data to a YAML file.

    Args:
        filepath (str): Path to the YAML file.
        calib_data (dict): Dictionary containing calibration data.
                           Expected keys: 'camera_matrix', 'dist_coeffs', 
                                          'image_width', 'image_height'.
                                          'reprojection_error' and 'num_valid_images' are for info.
    """
    data_to_save = {
        "image_width": calib_data["image_width"],
        "image_height": calib_data["image_height"],
        "camera_matrix": {
            "rows": calib_data["camera_matrix"].shape[0],
            "cols": calib_data["camera_matrix"].shape[1],
            "data": calib_data["camera_matrix"].flatten().tolist() # Flatten for YAML
        },
        "dist_coeffs": {
            "rows": calib_data["dist_coeffs"].shape[0],
            "cols": calib_data["dist_coeffs"].shape[1],
            "data": calib_data["dist_coeffs"].flatten().tolist() # Flatten for YAML
        },
        "reprojection_error": calib_data.get("reprojection_error", -1.0), # Optional but good to save
        "num_valid_images_for_calibration": calib_data.get("num_valid_images", 0) # Optional
    }
    
    file_path_obj = Path(filepath)
    file_path_obj.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    try:
        with open(file_path_obj, 'w') as f:
            yaml.dump(data_to_save, f, sort_keys=False, default_flow_style=None)
        print(f"Calibration results successfully saved to '{filepath}'.")
    except Exception as e:
        print(f"Error saving calibration results to '{filepath}': {e}")

def display_undistortion_preview(camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                                 image_dir: str, image_size: tuple[int, int]):
    """
    Displays a preview of undistortion on one of the calibration images.

    Args:
        camera_matrix (np.ndarray): The camera intrinsic matrix.
        dist_coeffs (np.ndarray): The distortion coefficients.
        image_dir (str): Directory where calibration images are stored.
        image_size (tuple[int, int]): (width, height) of the images.
    """
    image_files = list(Path(image_dir).glob("*.png")) + \
                  list(Path(image_dir).glob("*.jpg")) + \
                  list(Path(image_dir).glob("*.jpeg"))
    
    if not image_files:
        print("No images found in the directory to show undistortion preview.")
        return

    # Try to pick an image that was likely used in calibration (found corners)
    # This is a heuristic; ideally, we'd use one of the 'imgpoints' images.
    # For simplicity, just pick the first one or one from the middle.
    img_path_to_show = image_files[len(image_files) // 2] 
    
    print(f"Showing undistortion preview for: {img_path_to_show.name}")
    original_img = cv2.imread(str(img_path_to_show))
    if original_img is None:
        print(f"Could not load image {img_path_to_show.name} for preview.")
        return

    # Check if image dimensions match what calibration was done with
    h, w = original_img.shape[:2]
    if (w, h) != image_size:
        print(f"Warning: Image {img_path_to_show.name} dimensions ({w}x{h}) "
              f"differ from calibration image dimensions ({image_size[0]}x{image_size[1]}). "
              "Preview might be inaccurate if this image was not used or was resized.")
        # Attempt to resize to expected dimensions if it's different,
        # though ideally, user should select an image of the correct size.
        original_img = cv2.resize(original_img, image_size)

    # Get optimal new camera matrix for undistortion (optional, can help crop black areas)
    # alpha=0: results in an image with potentially black regions due to undistortion.
    # alpha=1: results in an image where all original pixels are retained, possibly with extra black areas.
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, alpha=0) #alpha=0 crops more
    
    # Undistort the image
    undistorted_img = cv2.undistort(original_img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop the image based on ROI (Region of Interest) from getOptimalNewCameraMatrix if alpha=0
    # x, y, w_roi, h_roi = roi
    # if alpha=0 was used and you want to crop to the valid pixel area:
    # undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi] 
    # However, for a simple side-by-side, not cropping might be better to show the effect.

    # Concatenate original and undistorted images for comparison
    # Ensure they have the same height for horizontal stacking
    if original_img.shape[0] != undistorted_img.shape[0] or original_img.shape[1] != undistorted_img.shape[1]:
        # This might happen if ROI cropping was aggressive or image sizes were mixed up.
        # Fallback to showing them separately or resizing one.
        # For now, just try to resize undistorted to original's shape for display
        undistorted_img_display = cv2.resize(undistorted_img, (original_img.shape[1], original_img.shape[0]))
    else:
        undistorted_img_display = undistorted_img

    comparison_img = np.concatenate((original_img, undistorted_img_display), axis=1)
    
    preview_window_name = "Undistortion Preview (Original vs. Undistorted) - Press any key to close"
    cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL) # Resizable
    
    # Calculate a good display size, e.g., fit to screen width
    screen_width_limit = 1280 
    current_comp_width = comparison_img.shape[1]
    if current_comp_width > screen_width_limit:
        scale_factor = screen_width_limit / current_comp_width
        disp_w = int(comparison_img.shape[1] * scale_factor)
        disp_h = int(comparison_img.shape[0] * scale_factor)
        cv2.resizeWindow(preview_window_name, disp_w, disp_h)

    cv2.imshow(preview_window_name, comparison_img)
    print("Displaying undistortion preview. Press any key in the preview window to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow(preview_window_name)


def get_checkerboard_dimensions(default_size: tuple[int, int]) -> tuple[int, int]:
    """
    Prompts the user for the inner dimensions of the checkerboard.

    Args:
        default_size (tuple[int, int]): Default (width, height) of inner corners.
                                       This is used if no command-line override is given.
    Returns:
        tuple[int, int]: The selected (width, height).
    """
    print("\n--- Checkerboard Dimensions Input ---")
    print("The checkerboard dimensions refer to the number of INNER corners.")
    print("For example, a standard board with 10x7 squares has 9x6 inner corners.")
    
    while True:
        try:
            user_input_w_str = input(f"Enter checkerboard width (number of inner corners, default: {default_size[0]}): ")
            width = int(user_input_w_str) if user_input_w_str else default_size[0]
            
            user_input_h_str = input(f"Enter checkerboard height (number of inner corners, default: {default_size[1]}): ")
            height = int(user_input_h_str) if user_input_h_str else default_size[1]

            if width > 1 and height > 1: # Checkerboard must have at least 2x2 inner corners
                return (width, height) # The print for confirmation is now in main_cli
            else:
                print("Dimensions must be positive integers greater than 1.")
        except ValueError:
            print("Invalid input. Please enter numbers for dimensions.")
        except Exception as e:
            print(f"An error occurred while parsing dimensions: {e}")
            # Fallback to default if unexpected error during input
            print(f"Falling back to default dimensions due to error: {default_size}")
            return default_size

def run_calibration(images_dir: str, checkerboard_dims: tuple[int, int]):
    """
    Performs camera calibration using collected images.

    Args:
        images_dir (str): Directory containing calibration images.
        checkerboard_dims (tuple[int, int]): (Width, Height) of inner checkerboard corners.

    Returns:
        dict | None: A dictionary with calibration results ('camera_matrix', 'dist_coeffs', 
                      'reprojection_error', 'image_width', 'image_height', 'num_valid_images') 
                      or None if calibration fails.
    """
    # Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(width-1,height-1,0)
    # These are 3D points in the checkerboard's own coordinate system.
    # checkerboard_dims[0] is width (number of inner corners along x-axis)
    # checkerboard_dims[1] is height (number of inner corners along y-axis)
    objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
    # Optional: If square size is known (e.g., in mm), scale objp: objp = objp * square_size_mm
    # For intrinsic calibration only, absolute scale of the board is not strictly needed.

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Use Path.glob to find images, more robust
    image_dir_path = Path(images_dir)
    image_files = list(image_dir_path.glob("*.png")) + \
                  list(image_dir_path.glob("*.jpg")) + \
                  list(image_dir_path.glob("*.jpeg"))
                  
    if not image_files:
        print(f"No .png, .jpg, or .jpeg images found in '{images_dir}'. Cannot calibrate.")
        return None
    
    print(f"\nFound {len(image_files)} images for calibration processing in '{images_dir}'.")
    
    # Initialize image dimensions. These will be updated by the first successfully processed image.
    img_width, img_height = 0, 0 
    valid_images_processed = 0

    for i, fname_path in enumerate(image_files):
        img = cv2.imread(str(fname_path))
        print(f"Processing image {i+1}/{len(image_files)}: {fname_path.name}...", end=" ")
        
        if img is None:
            print("Failed to load. Skipping.")
            continue
        
        current_h, current_w = img.shape[:2]
        if valid_images_processed == 0: # First successfully loaded image sets the expected dimensions
            img_width, img_height = current_w, current_h
            print(f"Dimensions set to {img_width}x{img_height}.", end=" ")
        elif (img_width != current_w) or (img_height != current_h):
            print(f"Mismatched dimensions ({current_w}x{current_h} vs expected {img_width}x{img_height}). Skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # Flags can sometimes help, e.g. cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)

        if ret:
            print("Checkerboard found.", end=" ")
            objpoints.append(objp)
            # Refine corner locations
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_subpix)
            valid_images_processed += 1 # Count images where checkerboard is found and dimensions match

            # Optional: Draw and display the corners (can be slow if many images)
            # cv2.drawChessboardCorners(img, checkerboard_dims, corners_subpix, ret)
            # cv2.imshow(f'Corners in {fname_path.name}', cv2.resize(img, (img_width//2, img_height//2)))
            # cv2.waitKey(200) 
        else:
            print("Checkerboard NOT found.")

    # cv2.destroyAllWindows() # If corners were being displayed

    if valid_images_processed == 0 : # No images where checkerboard was found
         print("\nNo checkerboards found in any of the processed images. Calibration cannot proceed.")
         return None

    if valid_images_processed < max(1, MIN_IMAGES_FOR_CALIBRATION // 2) : # Warn if very few successful detections
        print(f"\nWarning: Checkerboards successfully processed in only {valid_images_processed} images.")
        print(f"The recommended minimum is {MIN_IMAGES_FOR_CALIBRATION} images with detected checkerboards.")
        print("Calibration results may be inaccurate.")
        if input("Do you want to continue with calibration anyway? (y/n): ").lower() != 'y':
            print("Calibration aborted by user due to too few valid images.")
            return None

    print(f"\nCalibrating camera using {valid_images_processed} valid images (image size: {img_width}x{img_height})...")
    try:
        # Perform calibration
        # ret_cal: True if calibration is successful
        # mtx: Camera matrix (K)
        # dist: Distortion coefficients
        # rvecs: Rotation vectors for each view (one per valid image)
        # tvecs: Translation vectors for each view (one per valid image)
        ret_cal, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img_width, img_height), None, None)

        if not ret_cal:
            print("cv2.calibrateCamera returned False. Calibration failed.")
            return None

        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)): # Should be 'valid_images_processed' or len(rvecs)
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        reprojection_error = mean_error / len(objpoints) # Should be len(objpoints)

        return {
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "reprojection_error": reprojection_error,
            "image_width": img_width,
            "image_height": img_height,
            "num_valid_images": valid_images_processed # Number of images successfully used
        }

    except cv2.error as e:
        print(f"OpenCV Error during cv2.calibrateCamera: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during calibration: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_calibration_image_dir(img_dir: str, clear_existing: bool = True):
    """
    Creates the directory for calibration images.
    Optionally clears existing images if the directory already exists.

    Args:
        img_dir (str): The path to the directory for calibration images.
        clear_existing (bool): If True, removes existing files in the directory.
    """
    img_path = Path(img_dir)
    if img_path.exists():
        if clear_existing:
            print(f"Clearing existing images from '{img_dir}'...")
            for item in img_path.iterdir():
                if item.is_file():
                    item.unlink()
                # Optionally, could remove subdirectories too, but for now, only files.
    else:
        print(f"Creating directory for calibration images: '{img_dir}'")
        img_path.mkdir(parents=True, exist_ok=True)

def print_image_collection_instructions():
    """Prints instructions for collecting good calibration images."""
    print("\n--- Image Collection Guide ---")
    print("You will now collect images for camera calibration.")
    print("1. Print a checkerboard pattern (e.g., from a PDF online) and attach it to a flat, rigid surface.")
    print("2. Show the entire checkerboard to the camera from various angles and distances.")
    print("3. Ensure the checkerboard is well-lit and in focus.")
    print(f"4. Aim to capture {MIN_IMAGES_FOR_CALIBRATION}+ images for good results.")
    print("5. Vary the checkerboard's position: cover the center, edges, and corners of the camera's view.")
    print("6. Tilt the checkerboard in different orientations (forward, backward, left, right).")
    print("--- Preview Window Controls ---")
    print("  - Press 'c' or SPACEBAR to capture an image.")
    print("  - Press 'q' to finish collecting images.")
    print("---------------------------------")
    input("Press Enter to start image collection...")


def collect_calibration_images(camera_id: int, output_dir: str) -> int:
    """
    Opens a camera feed and allows the user to capture images for calibration.

    Args:
        camera_id (int): The ID of the camera to use.
        output_dir (str): Directory to save captured images.

    Returns:
        int: The number of images successfully captured.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera ID {camera_id}.")
        return 0

    img_count = 0
    window_name = "Calibration Image Capture - Press 'c' to capture, 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Make it resizable

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from camera.")
            break
        
        display_frame = frame.copy()
        # Display image count on the frame
        cv2.putText(display_frame, f"Images captured: {img_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'c' or SPACE to capture, 'q' to quit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Finished image collection.")
            break
        elif key == ord('c') or key == ord(' '): # Spacebar
            img_filename = Path(output_dir) / f"calibration_image_{img_count + 1:02d}.png"
            try:
                cv2.imwrite(str(img_filename), frame)
                img_count += 1
                print(f"Captured image {img_count}: {img_filename}")
            except Exception as e:
                print(f"Error saving image {img_filename}: {e}")
    
    cap.release()
    cv2.destroyAllWindows()
    return img_count


if __name__ == "__main__":
    # Ensure data directories exist (prepare_calibration_image_dir also does this for its specific dir)
    Path(CALIBRATION_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(CALIBRATION_RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Camera Calibration Assistant.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        "--board_dims",
        type=str,
        default=None, # Default to None, so we know if it was user-provided
        help="Inner dimensions of the checkerboard as 'width x height' (e.g., '9x6').\n"
             "If not provided, the script will prompt for these values or use a hardcoded default."
    )
    # Example for future extension:
    # parser.add_argument(
    #     "--min_images",
    #     type=int,
    #     default=MIN_IMAGES_FOR_CALIBRATION,
    #     help=f"Minimum number of calibration images to collect (default: {MIN_IMAGES_FOR_CALIBRATION})."
    # )
    args = parser.parse_args()

    # --- Process Arguments ---
    parsed_checkerboard_dims = None
    if args.board_dims:
        try:
            w_str, h_str = args.board_dims.lower().split('x')
            parsed_w = int(w_str)
            parsed_h = int(h_str)
            if parsed_w <= 1 or parsed_h <= 1:
                raise ValueError("Checkerboard dimensions (width and height) must both be greater than 1.")
            parsed_checkerboard_dims = (parsed_w, parsed_h)
        except ValueError as e:
            print(f"Error: Invalid --board_dims format ('{args.board_dims}'). Expected 'WIDTHxHEIGHT', e.g., '9x6'. {e}")
            print("Please correct the format or omit the argument to be prompted.")
            # Exit if critical argument is malformed
            exit(1) 
            # Alternatively, could fall back to prompting: parsed_checkerboard_dims = None

    # Call the main CLI function, passing the override if available
    main_cli(cli_checkerboard_dims_override=parsed_checkerboard_dims)
