import cv2
import numpy as np
import os

# Default checkerboard parameters
CHECKERBOARD_SIZE = (12, 8)  # (width, height) internal corners
SQUARE_SIZE_MM = 20.0
MAX_IMAGES = 15  # Maximum number of calibration images to capture
AUTO_CAPTURE_INTERVAL_SECONDS = 2.0 # Interval between automatic captures

# Directory to save calibration images (optional, but good for debugging)
CALIBRATION_IMAGE_DIR = "calibration_data"
MIN_CALIBRATION_IMAGES = 5


def capture_calibration_images(camera_index=0): # Added camera_index parameter
    """
    Captures images from the webcam for camera calibration automatically.

    Displays a live feed from the webcam. Images are captured automatically
    if checkerboard corners are detected and enough time has passed since
    the last capture. Pressing 'q' quits the capture process or signals
    calibration if max images are reached. Pressing ESC quits immediately.
    """
    if not os.path.exists(CALIBRATION_IMAGE_DIR):
        os.makedirs(CALIBRATION_IMAGE_DIR)
        print(f"Created directory: {CALIBRATION_IMAGE_DIR}")

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Prepare object points, like (0,0,0), (20,0,0), (40,0,0) ....,(220,140,0)
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

    cap = cv2.VideoCapture(camera_index) # Use camera_index

    if not cap.isOpened():
        print(f"Error: Could not open webcam with index {camera_index}.") # Info about index
        return [], [], None

    cv2.namedWindow("Calibration Feed")
    print(f"Hold checkerboard steady. Images will be captured automatically when detected.")
    print(f"{MAX_IMAGES} images are needed. Min {MIN_CALIBRATION_IMAGES} for calibration.")
    print("Press 'q' to quit (or proceed to calibrate if MAX_IMAGES reached).")
    print("Press ESC to discard and exit immediately.")

    img_count = 0
    capture_message = ""
    last_ui_message_time = 0 # For UI messages like "Max images captured"
    last_auto_capture_time = 0 # For timing automatic captures
    frame_size_for_return = None # Initialize frame_size to be returned

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_size_for_return is None: # Get it once
            frame_size_for_return = gray.shape[::-1] # (width, height)

        # Find the chess board corners
        corners_found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

        current_tick_count = cv2.getTickCount()

        # If found, draw corners and attempt automatic capture
        if corners_found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners_refined, corners_found)

            time_since_last_auto_capture = (current_tick_count - last_auto_capture_time) / cv2.getTickFrequency()

            if len(imgpoints) < MAX_IMAGES and time_since_last_auto_capture > AUTO_CAPTURE_INTERVAL_SECONDS:
                img_filename = os.path.join(CALIBRATION_IMAGE_DIR, f"calib_img_{len(imgpoints) + 1}.png")
                cv2.imwrite(img_filename, gray) # Save the gray image
                print(f"Saved {img_filename}")

                objpoints.append(objp)
                imgpoints.append(corners_refined)
                img_count = len(imgpoints)
                capture_message = f"Image {img_count}/{MAX_IMAGES} auto-captured."
                last_ui_message_time = current_tick_count # Update time for this message
                last_auto_capture_time = current_tick_count # Reset auto-capture timer
                print(capture_message)

                if img_count == MAX_IMAGES:
                    capture_message = f"Max images ({MAX_IMAGES}) captured. Press 'q' or wait."
                    print(capture_message)
                    last_ui_message_time = cv2.getTickCount() # Keep message displayed
            elif len(imgpoints) >= MAX_IMAGES:
                # This condition is to ensure the "Max images" message can persist if needed
                if not capture_message.startswith("Max images"): # Prevents spamming the same message
                    capture_message = "Max images already captured. Press 'q' or ESC."
                    last_ui_message_time = cv2.getTickCount()
        
        # Display image count
        text_to_display = f"Images: {len(imgpoints)} / {MAX_IMAGES}"
        cv2.putText(frame, text_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display capture_message (like "Image X captured" or "Max images")
        if capture_message and (current_tick_count - last_ui_message_time) / cv2.getTickFrequency() < 3.0: # Display message for 3 secs
            cv2.putText(frame, capture_message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            capture_message = "" # Clear message after timeout

        cv2.imshow("Calibration Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            print("ESC pressed. Discarding all captures and exiting.")
            objpoints = []
            imgpoints = []
            break
        elif key == 113:  # 'q' key
            if len(imgpoints) >= MIN_CALIBRATION_IMAGES:
                print(f"'q' pressed with {len(imgpoints)} images. Proceeding to calibration.")
                break
            elif len(imgpoints) >= MAX_IMAGES: # Should be caught by above, but as safety
                print("'q' pressed with max images. Proceeding to calibration.")
                break
            else:
                print(f"'q' pressed with only {len(imgpoints)} images (min {MIN_CALIBRATION_IMAGES} needed). Exiting capture mode without calibrating.")
                objpoints = [] # Discard points if not enough for calibration when quitting
                imgpoints = []
                break
        
        # Automatically proceed if MAX_IMAGES captured and some time passed (e.g. 5s) to allow 'q' or 'ESC'
        if len(imgpoints) >= MAX_IMAGES and \
           (current_tick_count - last_auto_capture_time) / cv2.getTickFrequency() > 5.0: # Wait 5s after last auto-capture at max
            if key != 27 and key != 113: # if user hasn't already pressed ESC or q
                print(f"Reached {MAX_IMAGES} images and timeout. Proceeding to calibration automatically.")
                break


    cap.release()
    cv2.destroyAllWindows()
    
    if len(objpoints) == 0 and len(imgpoints) == 0 and key != 27 : # If q was pressed early with no images.
        print("No calibration images were successfully captured or kept.")
    elif key == 27: # ESC was pressed
        print("Calibration discarded by user.")
        # objpoints and imgpoints are already cleared
    elif len(imgpoints) < MIN_CALIBRATION_IMAGES:
        print(f"Warning: Captured only {len(imgpoints)} images. Minimum {MIN_CALIBRATION_IMAGES} are required for calibration. Discarding.")
        objpoints = []
        imgpoints = []


    return objpoints, imgpoints, frame_size_for_return

if __name__ == '__main__':
    camera_idx_for_testing = 0 # Define a camera index for testing
    print(f"Running calibration capture for camera index: {camera_idx_for_testing}")
    objpoints, imgpoints, frame_size_from_capture = capture_calibration_images(camera_index=camera_idx_for_testing)

    if frame_size_from_capture:
        print(f"Frame size obtained from capture: {frame_size_from_capture}")
    else:
        if objpoints and imgpoints : # If points exist but frame size somehow None
             print("Warning: Frame size not obtained from capture, but points exist. Calibration might fail.")
        else: # No points, no frame size (e.g. early exit)
            print("Frame size not obtained from capture, as no valid image data was processed.")
        # Fallback for testing if absolutely necessary, but calibrate_camera will likely fail if frame_size is wrong
        # frame_size_from_capture = (640, 480) 

    if objpoints and imgpoints and len(imgpoints) >= MIN_CALIBRATION_IMAGES and frame_size_from_capture: #
        print(f"\nCaptured {len(imgpoints)} image sets.")
        print(f"Attempting to calibrate camera using frame size: {frame_size_from_capture}...")
        camera_matrix, dist_coeffs, reprojection_error = calibrate_camera(objpoints, imgpoints, frame_size_from_capture)
        if camera_matrix is not None and dist_coeffs is not None:
            print("\nCalibration successful.")
            
            # Construct filename based on camera_idx_for_testing
            filename_to_save = f"camera_params_idx{camera_idx_for_testing}.npz"
            calibration_file_path = os.path.join(CALIBRATION_IMAGE_DIR, filename_to_save)
            
            save_calibration_data(camera_matrix, dist_coeffs, frame_size_from_capture, reprojection_error, camera_index=camera_idx_for_testing) # Pass index
            
            print("\nAttempting to load calibration data back...")
            # Load using the same camera_idx_for_testing
            loaded_mtx, loaded_dist, loaded_fsize, loaded_r_error = load_calibration_data(camera_index=camera_idx_for_testing)
            if loaded_mtx is not None:
                print("Successfully loaded and verified saved data (see printed values above).")
        else:
            print("\nCalibration failed or was not performed due to insufficient data or missing frame size.")
    elif objpoints and imgpoints : # Captured some, but less than MIN_CALIBRATION_IMAGES or frame_size missing
        if not frame_size_from_capture:
            print(f"\nCaptured {len(imgpoints)} image sets, but frame size is missing. Cannot calibrate.")
        else: # frame_size is present, but not enough images
            print(f"\nCaptured {len(imgpoints)} image sets, but need at least {MIN_CALIBRATION_IMAGES} for calibration.")
    else:
        print("\nCalibration process was exited or no valid images were captured. Cannot calibrate.")


def calibrate_camera(objpoints, imgpoints, frame_size):
    """
    Performs camera calibration using the collected object and image points.

    Args:
        objpoints: List of 3D object points.
        imgpoints: List of 2D image points.
        frame_size: Tuple (width, height) of the images used for calibration.

    Returns:
        tuple: (camera_matrix, dist_coeffs, mean_reprojection_error) if calibration is successful,
               otherwise (None, None, None).
    """
    if not objpoints or not imgpoints:
        print("Error: Object points or image points list is empty. Cannot calibrate.")
        return None, None, None

    if len(imgpoints) < MIN_CALIBRATION_IMAGES: #
        print(f"Error: Insufficient number of calibration images. "
              f"Got {len(imgpoints)}, need at least {MIN_CALIBRATION_IMAGES}. Cannot calibrate.") #
        return None, None, None
    
    if frame_size is None or not (isinstance(frame_size, tuple) and len(frame_size) == 2):
        print(f"Error: Invalid frame_size: {frame_size}. Must be a tuple (width, height).")
        return None, None, None

    print(f"Starting calibration with {len(imgpoints)} images and frame size: {frame_size}...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None
    )

    if ret:
        print("\nCamera calibration successful!")
        print("Camera Matrix:\n", camera_matrix)
        print("\nDistortion Coefficients:\n", dist_coeffs)

        total_error = 0
        for i in range(len(objpoints)):
            projected_imgpoints, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(imgpoints[i], projected_imgpoints, cv2.NORM_L2) / len(projected_imgpoints)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        print(f"\nMean Reprojection Error: {mean_error}")
        
        return camera_matrix, dist_coeffs, mean_error
    else:
        print("\nError: Camera calibration failed.")
        return None, None, None


def save_calibration_data(camera_matrix, dist_coeffs, frame_size, reprojection_error, camera_index=None):
    """
    Saves camera calibration parameters to a .npz file.
    Filename is chosen based on camera_index.

    Args:
        camera_matrix (np.ndarray): The camera intrinsic matrix.
        dist_coeffs (np.ndarray): The distortion coefficients.
        frame_size (tuple): The frame size (width, height) used for calibration.
        reprojection_error (float): The mean reprojection error.
        camera_index (int, optional): Index of the camera. Defaults to None.
    """
    if camera_index is None:
        base_filename = "camera_params.npz"
    else:
        base_filename = f"camera_params_idx{camera_index}.npz"
    
    filepath = os.path.join(CALIBRATION_IMAGE_DIR, base_filename)

    try:
        if not os.path.exists(CALIBRATION_IMAGE_DIR): # Ensure base directory exists
            os.makedirs(CALIBRATION_IMAGE_DIR)
            print(f"Created directory: {CALIBRATION_IMAGE_DIR}")

        np.savez(filepath, 
                 camera_matrix=camera_matrix, 
                 dist_coeffs=dist_coeffs,
                 frame_size=np.array(frame_size), 
                 reprojection_error=reprojection_error)
        print(f"Calibration data saved to {filepath}")
    except Exception as e:
        print(f"Error saving calibration data to {filepath}: {e}")


def load_calibration_data(camera_index=None):
    """
    Loads camera calibration parameters from a .npz file based on camera_index.

    Args:
        camera_index (int, optional): Index of the camera. Defaults to None.

    Returns:
        tuple: (camera_matrix, dist_coeffs, frame_size, reprojection_error) if successful,
               otherwise (None, None, None, None).
    """
    if camera_index is None:
        base_filename = "camera_params.npz"
    else:
        base_filename = f"camera_params_idx{camera_index}.npz"
        
    filepath = os.path.join(CALIBRATION_IMAGE_DIR, base_filename)

    if not os.path.exists(filepath):
        print(f"Calibration file {filepath} not found.")
        return None, None, None, None

    try:
        data = np.load(filepath)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        frame_size = tuple(data['frame_size']) 
        reprojection_error = data['reprojection_error'].item() 

        print(f"Calibration data loaded from {filepath}")
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)
        print("Frame Size:", frame_size)
        print(f"Reprojection Error: {reprojection_error}")
        
        return camera_matrix, dist_coeffs, frame_size, reprojection_error
    except Exception as e:
        print(f"Error loading calibration data from {filepath}: {e}")
        return None, None, None, None
