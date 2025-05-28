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
from src.camera_selection import select_camera_interactive # New import


# CALIBRATION_FILE = os.path.join(CALIBRATION_IMAGE_DIR, "camera_params.npz") # No longer a single global file
MIN_MATCHES_FOR_POSE = 10 # Added constant

# Removed list_available_cameras function

def main():
    """
    Main function to manage camera calibration workflow.
    """
    # --- Camera Selection ---
    selected_camera_index = select_camera_interactive()

    if selected_camera_index is None:
        print("No camera selected or camera selection failed. Exiting application.")
        return
    
    # This print message is now handled within select_camera_interactive
    # print(f"Selected camera index: {selected_camera_index}") 

    camera_matrix = None
    dist_coeffs = None
    frame_size = None # To store the frame size used for calibration

    # Construct the expected calibration filename for the selected camera for user info
    # The actual loading relies on passing selected_camera_index to load_calibration_data
    expected_calibration_file = os.path.join(CALIBRATION_IMAGE_DIR, f"camera_params_idx{selected_camera_index}.npz")
    print(f"Using camera with index: {selected_camera_index}. Expecting calibration file: {expected_calibration_file}")
    
    # Attempt to load existing calibration data using selected_camera_index
    loaded_mtx, loaded_dist, loaded_fsize, loaded_r_error = load_calibration_data(camera_index=selected_camera_index)

    if loaded_mtx is not None and loaded_dist is not None and loaded_fsize is not None:
        print(f"Using existing calibration data for camera index {selected_camera_index}.")
        camera_matrix = loaded_mtx
        dist_coeffs = loaded_dist
        frame_size = loaded_fsize # Store the frame size
        print(f"  Reprojection error from loaded data: {loaded_r_error:.4f}")
    else:
        print(f"No existing calibration data found for camera index {selected_camera_index} or failed to load. Starting new calibration process.")
        
        # Call capture_calibration_images() with selected_camera_index
        # This now returns objpoints, imgpoints, frame_size_from_capture
        objpoints, imgpoints, frame_size_from_capture = capture_calibration_images(camera_index=selected_camera_index)

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

                    # Save the new calibration data using selected_camera_index
                    print(f"Saving new calibration data for camera index {selected_camera_index}...")
                    save_calibration_data(camera_matrix, dist_coeffs, frame_size, reprojection_error, camera_index=selected_camera_index)
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

    cap = cv2.VideoCapture(selected_camera_index) # Use selected camera
    if not cap.isOpened():
        print(f"Error: Could not open webcam with index {selected_camera_index} for feature tracking.")
        return

    cv2.namedWindow("Live Feed with Keypoints")
    cv2.namedWindow("Feature Matches")

    # --- 3D Visualization Setup ---
    viz_enabled = False
    viz_window = None
    placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8) # Define placeholder_img here for broader scope

    # Accumulated pose of the 'previous' camera in the world (starts at origin)
    world_R_previous = np.eye(3, dtype=np.float64)
    world_t_previous = np.zeros((3, 1), dtype=np.float64)

    # Pose of the 'current' camera in the world, to be displayed
    # This is initialized to the world origin and updated each frame where pose is valid
    display_R = np.eye(3, dtype=np.float64)
    display_t = np.zeros((3, 1), dtype=np.float64)

    try:
        import cv2.viz as viz
        viz_window = viz.Viz3d("3D Reconstruction")
        axis_widget = viz.WCoordinateSystem(scale=1.0) # Added scale for visibility
        viz_window.showWidget("CoordinateSystem", axis_widget)

        # Add a static camera widget for the world origin (first camera)
        origin_cam_pose_affine = viz.Affine3d(np.eye(3, dtype=np.float32), np.zeros((3,1), dtype=np.float32))
        origin_cam_widget = viz.WCameraPosition(0.5 * camera_matrix.astype(np.float32), scale=0.5, color=viz.Color.blue()) # Scale down intrinsic matrix if large
        viz_window.showWidget("OriginCamera", origin_cam_widget, origin_cam_pose_affine)
        
        # Initial camera pose widget (will be updated)
        # Ensure display_R and display_t are float32 for Affine3d
        initial_cam_pose_affine = viz.Affine3d(display_R.astype(np.float32), display_t.astype(np.float32))
        # Pass camera intrinsics (fx, fy, cx, cy) to WCameraPosition if available and scaled appropriately
        # For simplicity, we can use a generic representation or scale down the actual K matrix
        # Example: viz.WCameraPosition(0.5*K, scale=0.5) or viz.WCameraPosition(scale=0.5)
        # Using camera_matrix (K) obtained from calibration, ensure it's float32
        cam_widget = viz.WCameraPosition(0.5 * camera_matrix.astype(np.float32), scale=0.5, color=viz.Color.green())
        viz_window.showWidget("CurrentCameraPose", cam_widget, initial_cam_pose_affine)

        viz_enabled = True
        print("OpenCV Viz module loaded successfully. 3D Visualization enabled.")
    except ImportError:
        print("OpenCV Viz module (cv2.viz) not found. Install opencv-contrib-python for 3D visualization.")
        print("Falling back to placeholder 3D visualization.")
        cv2.namedWindow("3D Visualization Placeholder")
    except Exception as e:
        print(f"Error initializing OpenCV Viz: {e}")
        print("Falling back to placeholder 3D visualization.")
        cv2.namedWindow("3D Visualization Placeholder")

    # Initialize a variable to store the current camera pose affine transformation for Viz
    current_camera_affine_viz = viz.Affine3d(display_R.astype(np.float32), display_t.astype(np.float32)) if viz_enabled else None


    while True:
        if viz_enabled and viz_window and viz_window.wasStopped():
            print("3D Visualization window was closed by user.")
            break # Exit main loop if Viz window is closed

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
                    
                    # Store the pose of the previous camera (N-1) in the world, used for transforming points_3d
                    prev_cam_R_world = world_R_previous.copy()
                    prev_cam_t_world = world_t_previous.copy()
                    
                    # Accumulate pose: T_world_current = T_world_previous * T_previous_current
                    # R_est, t_est is T_previous_current (pose of current cam N relative to previous cam N-1)
                    # display_R, display_t will be T_world_current (pose of current cam N in world)
                    display_R = prev_cam_R_world @ R_est
                    display_t = prev_cam_t_world + prev_cam_R_world @ t_est

                    if viz_enabled and viz_window:
                        # Update the current camera pose widget in Viz
                        current_camera_affine_viz = viz.Affine3d(display_R.astype(np.float32), display_t.astype(np.float32))
                        viz_window.setWidgetPose("CurrentCameraPose", current_camera_affine_viz)
                    elif not viz_enabled:
                        # Placeholder: Print accumulated pose
                        rvec_display, _ = cv2.Rodrigues(display_R)
                        print(f"DEBUG: Accumulated Pose (Current Cam in World): R_vec={rvec_display.flatten()}, t={display_t.flatten()}")
                        # TODO: Visualize camera pose (display_R, display_t) in 3D window.
                    
                    # Define projection matrices for triangulation
                    # P1 is for the previous camera (N-1), in its own coordinate system [K|0]
                    # P2 is for the current camera (N), relative to camera (N-1) K@[R_est|t_est]
                    P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
                    P2 = camera_matrix @ np.hstack((R_est, t_est))
                    
                    # Triangulate points. These points are in the coordinate system of camera (N-1)
                    points_3d_relative_to_prev_cam = triangulate_points(points1, points2, P1, P2, inlier_mask=mask_pose)
                    
                    if points_3d_relative_to_prev_cam is not None and points_3d_relative_to_prev_cam.shape[0] > 0:
                        print(f"Reconstructed {points_3d_relative_to_prev_cam.shape[0]} 3D points.")
                        
                        # Transform points (which are relative to camera N-1) to world coordinates
                        # using the world pose of camera N-1 (prev_cam_R_world, prev_cam_t_world)
                        points_3d_world = (prev_cam_R_world @ points_3d_relative_to_prev_cam.T + prev_cam_t_world).T

                        if viz_enabled and viz_window:
                            points_3d_world_viz = points_3d_world.astype(np.float32)
                            cloud_widget = viz.WCloud(points_3d_world_viz, viz.Color.white())
                            viz_window.showWidget("point_cloud", cloud_widget)
                        elif not viz_enabled:
                            print(f"DEBUG: World 3D points for visualization: {points_3d_world[:5]}")
                            cv2.imshow("3D Visualization Placeholder", placeholder_img)
                            # TODO: Implement 3D visualization here. OpenCV Viz was problematic.
                    else:
                        print("Triangulation failed or yielded no 3D points.")
                        if viz_enabled and viz_window: 
                            try:
                                viz_window.removeWidget("point_cloud")
                            except: 
                                pass 
                        elif not viz_enabled:
                            cv2.imshow("3D Visualization Placeholder", placeholder_img) 

                    # Update world_R_previous and world_t_previous for the *next* iteration
                    # They become the pose of the current camera (N) in the world
                    world_R_previous = display_R.copy()
                    world_t_previous = display_t.copy()

                else: # R_est is None or t_est is None (pose estimation failed)
                    print("Pose estimation failed for this frame pair.")
                    # Do not update world_R_previous, world_t_previous here, keep last good pose.
                    # Also, display_R, display_t are not updated, so camera widget in Viz remains at last good pose.
                    if viz_enabled and viz_window: 
                        try:
                            viz_window.removeWidget("point_cloud") # No new points to show
                        except:
                            pass
                    elif not viz_enabled:
                        cv2.imshow("3D Visualization Placeholder", placeholder_img)
            else:
                print(f"Not enough good matches for pose estimation (found {len(good_matches)}, need {MIN_MATCHES_FOR_POSE}).")
            if viz_enabled and viz_window: # Clear previous cloud
                try:
                    viz_window.removeWidget("point_cloud")
                except:
                    pass
            elif not viz_enabled:
                cv2.imshow("3D Visualization Placeholder", placeholder_img)
        else:
            # Clear the matches window if no previous descriptors or current descriptors
            cv2.imshow("Feature Matches", np.zeros((480, 640, 3), dtype=np.uint8))
            if viz_enabled and viz_window: # Clear previous cloud
                try:
                    viz_window.removeWidget("point_cloud")
                except:
                    pass
            elif not viz_enabled:
                cv2.imshow("3D Visualization Placeholder", placeholder_img)


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

        if viz_enabled and viz_window:
            viz_window.spinOnce(1, True) # Refresh Viz window

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting feature tracking loop.")
            break

    # Release resources
    cap.release()
    if viz_enabled and viz_window:
        viz_window.close() # Close Viz window
    cv2.destroyAllWindows()
    print("Feature tracking loop finished.")


if __name__ == "__main__":
    main()
