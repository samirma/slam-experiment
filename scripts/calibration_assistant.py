import cv2
import os
import numpy as np
import yaml # Keep for loading/displaying if needed, but saving is delegated
from pathlib import Path
import argparse
import time

# Attempt to import the refactored calibration function
try:
    from scripts.calibrate_camera import perform_calibration_from_images, DEFAULT_SQUARE_SIZE_MM as CC_DEFAULT_SQUARE_SIZE_MM
except ImportError:
    print("Error: Could not import 'perform_calibration_from_images' from 'scripts.calibrate_camera'.")
    print("Please ensure 'calibrate_camera.py' is in the 'scripts' directory and your PYTHONPATH is set correctly.")
    # Fallback or exit if the core function cannot be imported
    # For simplicity, we'll let it fail later if the import doesn't work.
    # A more robust solution might involve adding parent dir to sys.path if running as script.
    # For now, assume it's run in an environment where 'scripts.calibrate_camera' is discoverable.
    # Define a placeholder if import fails to prevent immediate crash, error will occur at call time.
    def perform_calibration_from_images(*args, **kwargs):
        print("CRITICAL ERROR: perform_calibration_from_images was not imported correctly!")
        return None
    CC_DEFAULT_SQUARE_SIZE_MM = 20.0 # Fallback default

# Constants for the assistant
ASSISTANT_CALIBRATION_IMAGES_DIR = "data/calibration_images" # Where assistant saves images
ASSISTANT_CALIBRATION_RESULTS_FILE = "data/camera_calibration.yaml" # Where assistant tells calibration function to save
DEFAULT_ASSISTANT_CHECKERBOARD_SIZE = (12, 8) # (width, height) of inner corners
MIN_IMAGES_FOR_CALIBRATION = 10 # Recommended minimum by assistant

def list_available_cameras(max_cameras_to_check=10):
    available_cameras = []
    print("Detecting available cameras...")
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
            print(f"  Camera ID {i} is available.")
        else:
            print(f"  Camera ID {i} not accessible or does not exist.")
    if not available_cameras:
        print("No cameras found.")
    else:
        print(f"Found camera IDs: {available_cameras}")
    return available_cameras

def select_camera_id(available_cameras):
    if not available_cameras: return None
    while True:
        try:
            if len(available_cameras) == 1:
                print(f"Automatically selecting camera ID {available_cameras[0]}.")
                return available_cameras[0]
            user_input = input(f"Please enter camera ID from {available_cameras}: ")
            selected_id = int(user_input)
            if selected_id in available_cameras:
                print(f"Camera ID {selected_id} selected.")
                return selected_id
            else:
                print(f"Invalid ID. Please choose from {available_cameras}.")
        except ValueError: print("Invalid input. Please enter a number.")
        except Exception as e: print(f"An error occurred: {e}"); return None

def print_image_collection_instructions():
    print("\n--- Image Collection Guide ---")
    print("1. Print a checkerboard and attach to a flat, rigid surface.")
    print("2. Show the entire checkerboard from various angles and distances.")
    print("3. Ensure good lighting and focus.")
    print(f"4. Aim for {MIN_IMAGES_FOR_CALIBRATION}+ images.")
    print("5. Cover center, edges, and corners of camera view.")
    print("6. Tilt the board in different orientations.")
    print("--- Preview Window Controls ---")
    print("  - 'c' or SPACEBAR: Capture image.")
    print("  - 'a': Toggle Auto-Capture mode.")
    print("  - 'q': Finish collecting images.")
    print("---------------------------------")
    input("Press Enter to start image collection...")

def prepare_calibration_image_dir(img_dir: str, clear_existing: bool = True):
    img_path = Path(img_dir)
    if img_path.exists():
        if clear_existing:
            if input(f"Directory '{img_dir}' already exists. Clear existing images? (y/n): ").lower() == 'y':
                print(f"Clearing existing images from '{img_dir}'...")
                for item in img_path.iterdir():
                    if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        item.unlink()
            else:
                print(f"Using existing images in '{img_dir}' without clearing.")
    else:
        print(f"Creating directory for calibration images: '{img_dir}'")
        img_path.mkdir(parents=True, exist_ok=True)


def get_checkerboard_dimensions(default_size: tuple[int, int]) -> tuple[int, int]:
    print("\n--- Checkerboard Dimensions Input (Inner Corners) ---")
    while True:
        try:
            w_str = input(f"Enter checkerboard width (inner corners, default: {default_size[0]}): ")
            width = int(w_str) if w_str else default_size[0]
            h_str = input(f"Enter checkerboard height (inner corners, default: {default_size[1]}): ")
            height = int(h_str) if h_str else default_size[1]
            if width > 1 and height > 1:
                print(f"Using checkerboard inner dimensions: {width}x{height}")
                return (width, height)
            else: print("Dimensions must be > 1.")
        except ValueError: print("Invalid input. Enter numbers.")
        except Exception as e: print(f"Error: {e}"); return default_size

def get_square_size_mm(default_size_mm: float) -> float:
    print("\n--- Checkerboard Square Size Input ---")
    while True:
        try:
            size_str = input(f"Enter size of one checkerboard square in mm (default: {default_size_mm}): ")
            size = float(size_str) if size_str else default_size_mm
            if size > 0:
                print(f"Using square size: {size} mm")
                return size
            else: print("Square size must be positive.")
        except ValueError: print("Invalid input. Enter a number.")
        except Exception as e: print(f"Error: {e}"); return default_size_mm


def collect_calibration_images(camera_id: int, output_dir: str, checkerboard_dims: tuple[int, int]) -> int:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera ID {camera_id}.")
        return 0

    img_count = 0
    # Count existing images if not cleared
    existing_images = list(Path(output_dir).glob("calibration_image_*.png"))
    img_count = len(existing_images)
    print(f"Starting with {img_count} existing images in '{output_dir}'.")


    window_name = "Calibration Image Capture - 'c' capture, 'a' auto, 'q' quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    zone_names = ["Top-Left", "Top-Center", "Top-Right", "Mid-Left", "Mid-Center", "Mid-Right", "Bot-Left", "Bot-Center", "Bot-Right"]
    covered_zones = set()
    frame_height, frame_width = None, None
    auto_capture_mode = False
    last_capture_time = 0.0
    AUTO_CAPTURE_COOLDOWN_SECONDS = 2.0

    def _save_image_and_update_zones(current_frame: np.ndarray, current_img_idx: int, detected_corners, f_w: int, f_h: int) -> int:
        nonlocal covered_zones
        img_filename = Path(output_dir) / f"calibration_image_{current_img_idx + 1:03d}.png" # Use 3 digits
        try:
            cv2.imwrite(str(img_filename), current_frame)
            print(f"Captured image {current_img_idx + 1}: {img_filename.name}")
            if detected_corners is not None and f_w is not None and f_h is not None:
                centroid_x = np.mean(detected_corners[:,0,0]); centroid_y = np.mean(detected_corners[:,0,1])
                col = min(2, int(centroid_x / (f_w / 3))); row = min(2, int(centroid_y / (f_h / 3)))
                zone_name = zone_names[row * 3 + col]
                if zone_name not in covered_zones: covered_zones.add(zone_name); print(f"Zone: {zone_name} covered. Total: {len(covered_zones)}/{len(zone_names)}")
            return current_img_idx + 1
        except Exception as e: print(f"Error saving {img_filename.name}: {e}"); return current_img_idx

    while True:
        ret, frame = cap.read()
        if not ret: print("Error: Failed to grab frame."); break
        if frame_width is None: frame_height, frame_width = frame.shape[:2]

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)

        if ret_corners:
            cv2.drawChessboardCorners(display_frame, checkerboard_dims, corners, ret_corners)
            cv2.putText(display_frame, "Found!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "NOT Found", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display_frame, f"Images: {img_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Auto: {'ON' if auto_capture_mode else 'OFF'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if auto_capture_mode else (0,0,255), 2)
        
        needed_zones = [name for name in zone_names if name not in covered_zones]
        guidance_y = 90
        if not needed_zones:
            cv2.putText(display_frame, "All zones covered! Vary angles/dist or quit.", (10, guidance_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        else:
            cv2.putText(display_frame, "Need: " + ", ".join(needed_zones[:3]) + ("..." if len(needed_zones)>3 else ""), (10, guidance_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if auto_capture_mode and ret_corners and (time.time() - last_capture_time > AUTO_CAPTURE_COOLDOWN_SECONDS):
            print("Auto-capturing...")
            img_count = _save_image_and_update_zones(frame, img_count, corners, frame_width, frame_height)
            last_capture_time = time.time()

        if key == ord('q'): print("Finished collection."); break
        elif key == ord('a'): auto_capture_mode = not auto_capture_mode; print(f"Auto-capture: {auto_capture_mode}"); last_capture_time = time.time()
        elif key == ord('c') or key == ord(' '):
            img_count = _save_image_and_update_zones(frame, img_count, corners if ret_corners else None, frame_width, frame_height)

    cap.release(); cv2.destroyAllWindows(); return img_count


def display_undistortion_preview(camera_matrix_np: np.ndarray, dist_coeffs_np: np.ndarray,
                                 image_dir: str, image_size_wh: tuple[int, int]):
    image_files = list(Path(image_dir).glob("calibration_image_*.png")) # Prioritize saved images
    if not image_files: image_files = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.jpeg"))
    if not image_files: print("No images for undistortion preview."); return

    img_path = image_files[len(image_files) // 2]
    original_img = cv2.imread(str(img_path))
    if original_img is None: print(f"Could not load {img_path.name}."); return

    h, w = original_img.shape[:2]
    if (w, h) != image_size_wh:
        print(f"Warning: Preview image {img_path.name} ({w}x{h}) differs from calib size {image_size_wh}. Resizing.")
        original_img = cv2.resize(original_img, image_size_wh)

    new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix_np, dist_coeffs_np, image_size_wh, alpha=0.8) # Alpha 0.8 for less cropping
    undistorted_img = cv2.undistort(original_img, camera_matrix_np, dist_coeffs_np, None, new_cam_mtx)
    
    # Optional: crop with ROI
    # x_roi, y_roi, w_roi, h_roi = roi
    # undistorted_img = undistorted_img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    # original_img_roi = cv2.resize(original_img, (w_roi, h_roi)) # Resize original to match if cropping

    comparison_img = np.concatenate((original_img, undistorted_img), axis=1)
    win_name = "Undistortion Preview (Original | Undistorted) - Press any key"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    max_disp_w = 1600; current_comp_w = comparison_img.shape[1]
    if current_comp_w > max_disp_w:
        sf = max_disp_w / current_comp_w
        cv2.resizeWindow(win_name, int(comparison_img.shape[1]*sf), int(comparison_img.shape[0]*sf))
    
    cv2.imshow(win_name, comparison_img)
    print("Displaying undistortion preview. Press any key in the preview window...")
    cv2.waitKey(0); cv2.destroyWindow(win_name)


def main_assistant_cli(cli_checkerboard_dims_override: tuple[int, int] | None = None,
                       cli_square_size_override_mm: float | None = None):
    print("Welcome to the Camera Calibration Assistant!")
    available_cameras = list_available_cameras()
    if not available_cameras: print("Exiting: No cameras available."); return
    camera_id = select_camera_id(available_cameras)
    if camera_id is None: print("Exiting: No camera selected."); return

    print_image_collection_instructions()
    prepare_calibration_image_dir(ASSISTANT_CALIBRATION_IMAGES_DIR)

    checkerboard_dims = cli_checkerboard_dims_override if cli_checkerboard_dims_override else get_checkerboard_dimensions(DEFAULT_ASSISTANT_CHECKERBOARD_SIZE)
    square_size_mm = cli_square_size_override_mm if cli_square_size_override_mm else get_square_size_mm(CC_DEFAULT_SQUARE_SIZE_MM)

    num_collected = collect_calibration_images(camera_id, ASSISTANT_CALIBRATION_IMAGES_DIR, checkerboard_dims)

    if num_collected == 0: print("\nNo images collected. Cannot calibrate."); return
    if num_collected < MIN_IMAGES_FOR_CALIBRATION:
        print(f"\nWarning: Only {num_collected} images. Recommended: {MIN_IMAGES_FOR_CALIBRATION}.")
        if input("Continue calibration? (y/n): ").lower() != 'y': print("Calibration aborted."); return

    print(f"\nAttempting calibration using images from '{ASSISTANT_CALIBRATION_IMAGES_DIR}'...")
    # Call the imported calibration function
    calibration_results_dict = perform_calibration_from_images(
        images_dir_path=ASSISTANT_CALIBRATION_IMAGES_DIR,
        checkerboard_dims=checkerboard_dims,
        square_size_mm=square_size_mm,
        output_yaml_filepath=ASSISTANT_CALIBRATION_RESULTS_FILE
    )

    if calibration_results_dict:
        print("\nCalibration process finished by 'perform_calibration_from_images'.")
        # Results (K, dist, error) are already printed by perform_calibration_from_images
        # The YAML file is also saved by it.

        if input("\nView undistortion preview? (y/n): ").lower() == 'y':
            # Ensure K and dist_coeffs are numpy arrays for cv2 functions
            k_matrix_np = np.array(calibration_results_dict['camera_matrix'])
            dist_coeffs_np = np.array(calibration_results_dict['dist_coeffs'])
            img_w = calibration_results_dict['image_width']
            img_h = calibration_results_dict['image_height']
            
            display_undistortion_preview(
                k_matrix_np,
                dist_coeffs_np,
                ASSISTANT_CALIBRATION_IMAGES_DIR,
                (img_w, img_h)
            )
        print(f"\nCalibration complete. Results saved to '{ASSISTANT_CALIBRATION_RESULTS_FILE}'.")
    else:
        print("\nCalibration failed as reported by 'perform_calibration_from_images'.")

if __name__ == "__main__":
    Path(ASSISTANT_CALIBRATION_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(ASSISTANT_CALIBRATION_RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Interactive Camera Calibration Assistant.")
    parser.add_argument("--board_dims", type=str, default=None, help="Checkerboard inner dimensions 'WIDTHxHEIGHT' (e.g., '9x6').")
    parser.add_argument("--square_size", type=float, default=None, help="Size of one checkerboard square in mm.")
    args = parser.parse_args()

    parsed_dims = None
    if args.board_dims:
        try:
            w, h = map(int, args.board_dims.lower().split('x'))
            if w <= 1 or h <= 1: raise ValueError("Dims > 1.")
            parsed_dims = (w, h)
        except ValueError as e: print(f"Error: Invalid --board_dims ('{args.board_dims}'). {e}. Using prompt/default."); exit(1)
    
    parsed_square_size = args.square_size if args.square_size and args.square_size > 0 else None
    if args.square_size and (parsed_square_size is None): # Only print error if user provided invalid
        print(f"Error: Invalid --square_size ('{args.square_size}'). Must be positive. Using prompt/default.")
        # No exit, will prompt

    main_assistant_cli(cli_checkerboard_dims_override=parsed_dims, 
                       cli_square_size_override_mm=parsed_square_size)
