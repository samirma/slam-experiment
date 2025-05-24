import os
from src.calibration import (
    load_calibration_data,
    capture_calibration_images,
    calibrate_camera,
    save_calibration_data,
    MIN_CALIBRATION_IMAGES, # Useful for checking if enough images were captured
    CALIBRATION_IMAGE_DIR, # For constructing default path
    MIN_CALIBRATION_IMAGES # For checking if enough images were captured
)
from src.feature_utils import detect_features, match_features
from src.sfm import estimate_pose, triangulate_points # Added
import cv2
import numpy as np


CALIBRATION_FILE = os.path.join(CALIBRATION_IMAGE_DIR, "camera_params.npz")
MIN_MATCHES_FOR_POSE = 10 # Added constant

def main():
    """
    Main function to manage camera calibration workflow.
    """
    camera_matrix = None
    dist_coeffs = None
    frame_size = None # To store the frame size used for calibration

    print(f"Attempting to load calibration data from: {CALIBRATION_FILE}")
    # Attempt to load existing calibration data
    loaded_mtx, loaded_dist, loaded_fsize, loaded_r_error = load_calibration_data(CALIBRATION_FILE)

    if loaded_mtx is not None and loaded_dist is not None and loaded_fsize is not None:
        print("Using existing calibration data.")
        camera_matrix = loaded_mtx
        dist_coeffs = loaded_dist
        frame_size = loaded_fsize # Store the frame size
        print(f"  Reprojection error from loaded data: {loaded_r_error:.4f}")
    else:
        print("No existing calibration data found or failed to load. Starting new calibration process.")
        
        # Call capture_calibration_images()
        # This now returns objpoints, imgpoints, frame_size_from_capture
        objpoints, imgpoints, frame_size_from_capture = capture_calibration_images()

        if frame_size_from_capture is None and objpoints and imgpoints:
            # This case should ideally not happen if capture_calibration_images always returns a frame_size
            # when images are present. But as a fallback:
            print("Warning: Frame size not returned from capture, but images were captured. This is unexpected.")
            # Attempt to determine frame size from the first saved image if possible (concept)
            # For now, we'll rely on capture_calibration_images to provide it.
            # If it's critical and not provided, calibration might fail or use wrong defaults.

        if objpoints and imgpoints and len(imgpoints) >= MIN_CALIBRATION_IMAGES:
            print(f"Captured {len(imgpoints)} images for calibration.")
            if frame_size_from_capture:
                print(f"Frame size from capture: {frame_size_from_capture}")
                # Calibrate the camera
                # calibrate_camera returns camera_matrix, dist_coeffs, reprojection_error
                new_mtx, new_dist, reprojection_error = calibrate_camera(objpoints, imgpoints, frame_size_from_capture)

                if new_mtx is not None and new_dist is not None:
                    print("Calibration successful.")
                    camera_matrix = new_mtx
                    dist_coeffs = new_dist
                    frame_size = frame_size_from_capture # Store the frame size used

                    # Save the new calibration data
                    print(f"Saving new calibration data to {CALIBRATION_FILE}...")
                    save_calibration_data(CALIBRATION_FILE, camera_matrix, dist_coeffs, frame_size, reprojection_error)
                else:
                    print("Calibration failed. Using default/no calibration.")
                    # camera_matrix, dist_coeffs remain None
            else:
                print("Error: Frame size from capture is missing. Cannot proceed with calibration.")
                # camera_matrix, dist_coeffs remain None
        
        elif objpoints and imgpoints: # Not enough images
             print(f"Only {len(imgpoints)} images captured, need at least {MIN_CALIBRATION_IMAGES}. Using default/no calibration.")
             # camera_matrix, dist_coeffs remain None
        else: # No images captured or user quit early
            print("No images captured for calibration. Using default/no calibration.")
            # camera_matrix, dist_coeffs remain None

    # Final status print
    if camera_matrix is not None and dist_coeffs is not None:
        print("\n--- Calibration Status ---")
        print("Camera calibrated successfully.")
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)
        if frame_size:
             print("Frame Size used for calibration:", frame_size)
        # Reprojection error would have been printed by load or calibrate functions
    else:
        print("\n--- Calibration Status ---")
        print("Camera not calibrated. Reconstruction may be inaccurate or fail.")
        print("Cannot start feature tracking without camera calibration.")
        return # Exit if not calibrated

    # --- Feature Detection and Matching Loop ---
    print("\nStarting feature detection and matching loop...")
    
    prev_keypoints = None
    prev_descriptors = None
    prev_undistorted_color_frame_for_drawing = None # To store the previous undistorted color frame

    orb_detector = cv2.ORB_create() # Initialize ORB detector once

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam for feature tracking.")
        return

    cv2.namedWindow("Live Feed with Keypoints")
    cv2.namedWindow("Feature Matches")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting feature tracking loop.")
            break

        # Undistort both color and grayscale frames
        undistorted_color_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, None)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Original gray for detection
        undistorted_gray_frame = cv2.undistort(gray_frame, camera_matrix, dist_coeffs, None, None)
        
        # Detect features
        current_keypoints, current_descriptors = detect_features(
            undistorted_gray_frame, detector=orb_detector
        )

        # Match features if previous descriptors exist
        if prev_descriptors is not None and current_descriptors is not None and len(current_descriptors) > 0 and prev_keypoints is not None:
            good_matches = match_features(prev_descriptors, current_descriptors, detector_type='orb')
            
            # Draw matches for visualization
            if good_matches and prev_undistorted_color_frame_for_drawing is not None:
                img_matches_display = cv2.drawMatches(
                    prev_undistorted_color_frame_for_drawing, prev_keypoints,
                    undistorted_color_frame, current_keypoints,
                    good_matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imshow("Feature Matches", img_matches_display)
            else: 
                # Clear the matches window if no good_matches or no prev frame for drawing
                cv2.imshow("Feature Matches", np.zeros((480, 640, 3), dtype=np.uint8))

            # --- Two-View Reconstruction ---
            if len(good_matches) >= MIN_MATCHES_FOR_POSE:
                # Extract 2D point coordinates for pose estimation
                points1 = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                points2 = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

                # Estimate pose
                # dist_coeffs is used by estimate_pose if points were not undistorted prior to findEssentialMat
                # However, our points1 and points2 are from undistorted_gray_frame features,
                # so they are already "undistorted" in that sense.
                # The estimate_pose function itself passes dist_coeffs=None to cv2.findEssentialMat and cv2.recoverPose
                # assuming the input points (points1, points2) are already corrected for lens distortion.
                R_est, t_est, E_est, mask_pose = estimate_pose(points1, points2, camera_matrix, dist_coeffs=None)

                if R_est is not None and t_est is not None:
                    print("Pose estimated successfully for this frame pair.")
                    
                    # Define projection matrices
                    P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
                    P2 = camera_matrix @ np.hstack((R_est, t_est))
                    
                    # Triangulate points
                    # The mask_pose from estimate_pose (which is from recoverPose) is applicable to points1 and points2
                    points_3d = triangulate_points(points1, points2, P1, P2, inlier_mask=mask_pose)
                    
                    if points_3d is not None and points_3d.shape[0] > 0:
                        print(f"Reconstructed {points_3d.shape[0]} 3D points.")
                        # For now, just printing. Future steps might involve storing/visualizing these.
                    else:
                        print("Triangulation failed or yielded no 3D points.")
                else:
                    print("Pose estimation failed for this frame pair.")
            else:
                print(f"Not enough good matches for pose estimation (found {len(good_matches)}, need {MIN_MATCHES_FOR_POSE}).")
        else:
            # Clear the matches window if no previous descriptors or current descriptors
             cv2.imshow("Feature Matches", np.zeros((480, 640, 3), dtype=np.uint8))


        # Draw keypoints on the current undistorted color frame
        if current_keypoints:
            img_keypoints_live = cv2.drawKeypoints(
                undistorted_color_frame.copy(), current_keypoints, None, color=(0, 255, 0)
            )
            cv2.imshow("Live Feed with Keypoints", img_keypoints_live)
        else:
            cv2.imshow("Live Feed with Keypoints", undistorted_color_frame) # Show undistorted if no keypoints

        # Update previous state
        prev_keypoints = current_keypoints
        prev_descriptors = current_descriptors
        prev_undistorted_color_frame_for_drawing = undistorted_color_frame.copy()


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting feature tracking loop.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Feature tracking loop finished.")


if __name__ == "__main__":
    main()
