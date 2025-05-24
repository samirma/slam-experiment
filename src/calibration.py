import cv2
import numpy as np
import os

# Default checkerboard parameters
CHECKERBOARD_SIZE = (12, 8)  # (width, height) internal corners
SQUARE_SIZE_MM = 20.0
MAX_IMAGES = 15  # Maximum number of calibration images to capture

# Directory to save calibration images (optional, but good for debugging)
CALIBRATION_IMAGE_DIR = "calibration_data"
MIN_CALIBRATION_IMAGES = 5


def capture_calibration_images():
    """
    Captures images from the webcam for camera calibration.

    Displays a live feed from the webcam. Pressing the spacebar triggers an
    image capture if checkerboard corners are detected. Pressing 'q' quits
    the capture process or signals calibration if max images are reached.
    Pressing ESC quits immediately.
    """
    if not os.path.exists(CALIBRATION_IMAGE_DIR):
        os.makedirs(CALIBRATION_IMAGE_DIR)
        print(f"Created directory: {CALIBRATION_IMAGE_DIR}")

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Prepare object points, like (0,0,0), (20,0,0), (40,0,0) ....,(220,140,0)
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return [], [], None

    cv2.namedWindow("Calibration Feed")
    print(f"Press SPACE to capture an image ({MAX_IMAGES} needed).")
    print("Press 'q' to quit (or proceed to calibrate if MAX_IMAGES reached).")
    print("Press ESC to discard and exit immediately.")

    img_count = 0
    capture_message = ""
    last_capture_time = 0
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

        # If found, add object points, image points (after refining them)
        if corners_found:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners_refined, corners_found)
        
        # Display image count
        text_to_display = f"Images: {len(imgpoints)} / {MAX_IMAGES}"
        cv2.putText(frame, text_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if capture_message and (cv2.getTickCount() - last_capture_time) / cv2.getTickFrequency() < 2.0: # Display message for 2 secs
            cv2.putText(frame, capture_message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            capture_message = ""

        cv2.imshow("Calibration Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            print("ESC pressed. Discarding all captures and exiting.")
            objpoints = []
            imgpoints = []
            break
        elif key == 32:  # Spacebar
            if corners_found:
                if len(imgpoints) < MAX_IMAGES:
                    # Save image (optional, for debugging or manual inspection)
                    img_filename = os.path.join(CALIBRATION_IMAGE_DIR, f"calib_img_{len(imgpoints) + 1}.png")
                    cv2.imwrite(img_filename, gray) # Save the gray image as corners are found on it
                    print(f"Saved {img_filename}")

                    objpoints.append(objp)
                    imgpoints.append(corners_refined)
                    img_count = len(imgpoints)
                    capture_message = f"Image {img_count}/{MAX_IMAGES} captured."
                    last_capture_time = cv2.getTickCount()
                    print(capture_message)

                    if img_count == MAX_IMAGES:
                        capture_message = f"Max images ({MAX_IMAGES}) captured. Press 'q' to calibrate or ESC to discard."
                        print(capture_message)
                        last_capture_time = cv2.getTickCount() # Keep message displayed
                else:
                    capture_message = "Max images already captured. Press 'q' or ESC."
                    last_capture_time = cv2.getTickCount()
            else:
                capture_message = "No checkerboard found. Try different angles."
                last_capture_time = cv2.getTickCount()

        elif key == 113:  # 'q' key
            if len(imgpoints) >= MAX_IMAGES:
                print("'q' pressed with max images. Proceeding to calibration (conceptually).")
                break 
            else:
                print(f"'q' pressed before capturing {MAX_IMAGES} images. Exiting capture mode.")
                # Optionally, clear points if quitting means discarding partial progress
                # objpoints = [] 
                # imgpoints = []
                break
        
        # Check if max images captured and then user pressed 'q' implicitly by loop condition
        if len(imgpoints) >= MAX_IMAGES and key == 113: # Already handled above, but as a safety
             break


    cap.release()
    cv2.destroyAllWindows()
    
    if len(imgpoints) < MAX_IMAGES and key != 27: # if not ESC and not enough images
        print(f"Warning: Captured only {len(imgpoints)} out of {MAX_IMAGES} required images.")
        # Decide if partial data should be returned or cleared
        # return [], [] # Option: clear if not enough

    if not imgpoints: # if imgpoints is empty (either due to ESC or quitting early)
        print("No calibration images were successfully captured.")

    return objpoints, imgpoints, frame_size_for_return

if __name__ == '__main__':
    # This is for testing purposes
    # capture_calibration_images now also returns frame_size
    
    objpoints, imgpoints, frame_size_from_capture = capture_calibration_images()

    # Use frame_size_from_capture if available, otherwise fallback
    if frame_size_from_capture:
        print(f"Frame size obtained from capture: {frame_size_from_capture}")
    else:
        # Fallback if frame_size_from_capture is None (e.g., no frames processed)
        print("Frame size not obtained from capture, using default (640, 480) for test calibration.")
        frame_size_from_capture = (640, 480) # Default for testing if none came back

    if objpoints and imgpoints and len(imgpoints) >= MIN_CALIBRATION_IMAGES:
        print(f"\nCaptured {len(imgpoints)} image sets.")
        print(f"Attempting to calibrate camera using frame size: {frame_size_from_capture}...")
        # Note: calibrate_camera now returns reprojection_error as the third value
        camera_matrix, dist_coeffs, reprojection_error = calibrate_camera(objpoints, imgpoints, frame_size_from_capture)
        if camera_matrix is not None and dist_coeffs is not None:
            print("\nCalibration successful.")
            
            # Save the calibration data
            calibration_file = os.path.join(CALIBRATION_IMAGE_DIR, "camera_params.npz")
            save_calibration_data(calibration_file, camera_matrix, dist_coeffs, frame_size_from_capture, reprojection_error)
            
            # Attempt to load it back (for testing)
            print("\nAttempting to load calibration data back...")
            loaded_mtx, loaded_dist, loaded_fsize, loaded_r_error = load_calibration_data(calibration_file)
            if loaded_mtx is not None:
                print("Successfully loaded and verified saved data (see printed values above).")
        else:
            print("\nCalibration failed or was not performed due to insufficient data.")
    elif objpoints and imgpoints: # Captured some, but less than MIN_CALIBRATION_IMAGES
        print(f"\nCaptured {len(imgpoints)} image sets, but need at least {MIN_CALIBRATION_IMAGES} for calibration.")
    else:
        print("\nCalibration process was exited or no images were captured. Cannot calibrate.")


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

    if len(imgpoints) < MIN_CALIBRATION_IMAGES:
        print(f"Error: Insufficient number of calibration images. "
              f"Got {len(imgpoints)}, need at least {MIN_CALIBRATION_IMAGES}. Cannot calibrate.")
        return None, None, None
    
    if frame_size is None or not (isinstance(frame_size, tuple) and len(frame_size) == 2):
        print(f"Error: Invalid frame_size: {frame_size}. Must be a tuple (width, height).")
        return None, None, None

    print(f"Starting calibration with {len(imgpoints)} images and frame size: {frame_size}...")

    # cv2.calibrateCamera returns: ret, camera_matrix, dist_coeffs, rvecs, tvecs
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None
    )

    if ret:
        print("\nCamera calibration successful!")
        print("Camera Matrix:\n", camera_matrix)
        print("\nDistortion Coefficients:\n", dist_coeffs)

        # Calculate and print reprojection error
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


def save_calibration_data(filename, camera_matrix, dist_coeffs, frame_size, reprojection_error):
    """
    Saves camera calibration parameters to a .npz file.

    Args:
        filename (str): Path to the file where data will be saved.
        camera_matrix (np.ndarray): The camera intrinsic matrix.
        dist_coeffs (np.ndarray): The distortion coefficients.
        frame_size (tuple): The frame size (width, height) used for calibration.
        reprojection_error (float): The mean reprojection error.
    """
    try:
        # Ensure the directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        np.savez(filename, 
                 camera_matrix=camera_matrix, 
                 dist_coeffs=dist_coeffs,
                 frame_size=np.array(frame_size), # Ensure frame_size is an ndarray for saving
                 reprojection_error=reprojection_error)
        print(f"Calibration data saved to {filename}")
    except Exception as e:
        print(f"Error saving calibration data to {filename}: {e}")


def load_calibration_data(filename):
    """
    Loads camera calibration parameters from a .npz file.

    Args:
        filename (str): Path to the file from which data will be loaded.

    Returns:
        tuple: (camera_matrix, dist_coeffs, frame_size, reprojection_error) if successful,
               otherwise (None, None, None, None).
    """
    if not os.path.exists(filename):
        print(f"Calibration file {filename} not found.")
        return None, None, None, None

    try:
        data = np.load(filename)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        # frame_size was saved as np.array, convert back to tuple if needed by other functions
        frame_size = tuple(data['frame_size']) 
        reprojection_error = data['reprojection_error'].item() # .item() if it's a 0-dim array

        print(f"Calibration data loaded from {filename}")
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)
        print("Frame Size:", frame_size)
        print(f"Reprojection Error: {reprojection_error}")
        
        return camera_matrix, dist_coeffs, frame_size, reprojection_error
    except Exception as e:
        print(f"Error loading calibration data from {filename}: {e}")
        return None, None, None, None
