import os
from src.calibration import (
    load_calibration_data,
    capture_calibration_images,
    calibrate_camera,
    save_calibration_data,
    MIN_CALIBRATION_IMAGES, # Useful for checking if enough images were captured
    CALIBRATION_IMAGE_DIR, # For constructing default path
    MIN_CALIBRATION_IMAGES, # For checking if enough images were captured
    save_calibration_for_pyslam # Added import
)
# from src.feature_utils import detect_features, match_features # Removed: pyslam handles features
# from src.sfm import estimate_pose, triangulate_points # Removed: pyslam handles SfM/SLAM
import cv2
import numpy as np
from src.camera_selection import select_camera_interactive # New import
import open3d as o3d # Removed: pyslam handles visualization

import yaml # Added for pyslam config
import time # Added for timestamps for pyslam

# --- Placeholder for pyslam imports ---
# NOTE: These are speculative and depend on the actual pyslam library structure.
import pyslam
from pyslam.config import Config as PyslamConfig
from pyslam.slam_system import SLAMSystem
# --- End Placeholder for pyslam imports ---


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

        # Save calibration for pyslam
        save_calibration_for_pyslam(camera_matrix, dist_coeffs, frame_size, selected_camera_index)
        # The function save_calibration_for_pyslam already prints a message,
        # so an additional print here might be redundant unless more specific context is needed.
        # print(f"pyslam camera settings saved to config/pyslam_settings_idx{selected_camera_index}.yaml")

        # --- Initialize Pyslam (Placeholder) ---
        print("\n--- Initializing pyslam System ---")
        slam_system = None # Placeholder for the SLAM system object
        pyslam_config_path = "config/pyslam_config.yaml"

        try:
            with open(pyslam_config_path, 'r') as f:
                pyslam_general_config_data = yaml.safe_load(f)

            # Dynamically update the path to the camera-specific settings file
            camera_settings_file = f"config/pyslam_settings_idx{selected_camera_index}.yaml"
            pyslam_general_config_data['DATASET']['FOLDER_DATASET']['settings_file'] = camera_settings_file

            print(f"Updated pyslam config to use camera settings: {camera_settings_file}")

            # --- Speculative pyslam Initialization ---
            # NOTE: The following lines are highly speculative and depend on pyslam's API.
            print("Initializing Pyslam with the following configuration data:") # New print for debugging
            print(yaml.dump(pyslam_general_config_data)) # New print for debugging

            # Ensure pyslam.config and pyslam.slam_system are imported
            # For example, add near other imports:
            # import pyslam
            # from pyslam.config import Config as PyslamConfig
            # from pyslam.slam_system import SLAMSystem

            pyslam_config_obj = PyslamConfig(config_dict=pyslam_general_config_data)
            slam_system = SLAMSystem(config=pyslam_config_obj,
                                       enable_viewer=pyslam_config_obj.get_global_param('kUseViewer', True))
            print("pyslam system initialized.") # Modified print
            # --- End Speculative pyslam Initialization ---

            # For now, we'll just print that we would initialize it.
            # print(f"INFO: Pyslam would be initialized here using the main config '{pyslam_config_path}' and camera-specific settings '{camera_settings_file}'.")
            # print("INFO: Actual pyslam initialization code is commented out as its API is not yet known.")
            # Set a dummy slam_system if actual pyslam is not being run to avoid errors later if any logic uses it.
            # For this subtask, we are commenting out processing, so it might not be strictly needed.
            # slam_system = "dummy_pyslam_object" # This line is effectively replaced by the actual init

        except Exception as e:
            print(f"Error loading or modifying pyslam configuration: {e}")
            print("Cannot proceed with pyslam. Exiting.")
            return

    else:
        print("\n--- Calibration Status ---")
        print("Camera not calibrated. Reconstruction may be inaccurate or fail.")
        print("Cannot start feature tracking without camera calibration.")
        return # Exit if not calibrated

    # --- Pyslam Processing Loop ---
    print("\nStarting pyslam processing loop...")
    
    # prev_keypoints = None # Commented out for pyslam
    # prev_descriptors = None # Commented out for pyslam
    # prev_undistorted_color_frame_for_drawing = None # Commented out for pyslam
    # orb_detector = cv2.ORB_create() # Commented out for pyslam

    cap = cv2.VideoCapture(selected_camera_index) # Use selected camera
    if not cap.isOpened():
        print(f"Error: Could not open webcam with index {selected_camera_index} for pyslam.")
        return

    cv2.namedWindow("Live Feed to pyslam") # Renamed window
    # cv2.namedWindow("Feature Matches") # Commented out for pyslam

    # --- Open3D Visualization Setup ---
    print("Initializing Open3D visualizer...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Reconstruction")
    point_cloud_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(point_cloud_o3d)
    world_origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(world_origin_axes)
    canonical_camera_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    current_cam_vis_o3d = o3d.geometry.TriangleMesh()
    current_cam_vis_o3d.vertices = canonical_camera_axes.vertices
    current_cam_vis_o3d.triangles = canonical_camera_axes.triangles
    current_cam_vis_o3d.compute_vertex_normals()
    current_cam_vis_o3d.paint_uniform_color([0.1, 0.1, 0.7]) # Blue color for camera
    vis.add_geometry(current_cam_vis_o3d)
    view_control = vis.get_view_control()
    if view_control:
        view_control.set_zoom(0.8)
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, -1, 0])
        view_control.set_front([0, 0, -1])
    else:
        print("[WARNING] Failed to get Open3D view control...")
    # world_R_previous = np.eye(3, dtype=np.float64) # Commented out - SLAM system manages pose
    # world_t_previous = np.zeros((3, 1), dtype=np.float64) # Commented out - SLAM system manages pose
    # display_R = np.eye(3, dtype=np.float64) # Commented out - SLAM system manages pose
    # display_t = np.zeros((3, 1), dtype=np.float64) # Commented out - SLAM system manages pose

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting processing loop.")
            break

        # Undistort color frame (pyslam might do this internally if raw images are preferred,
        # but often SLAM systems expect undistorted images or handle distortion via calibration file)
        undistorted_color_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, None)
        
        current_timestamp = time.time()

        # --- Pyslam Frame Processing (Placeholder) ---
        if slam_system is not None:
            try:
                # NOTE: This is a speculative call to pyslam's process_image method.
                # The actual method signature (e.g., needs BGR or RGB, specific timestamp format)
                # needs to be verified from pyslam's documentation.
                # Example:
                slam_system.process_image(undistorted_color_frame, timestamp=current_timestamp)

                # For now, just indicate it would be processed.
                # print(f"Timestamp: {current_timestamp:.3f} - Frame processed by pyslam (Placeholder).")
                # pass # Placeholder for actual call # Removed

            except Exception as e:
                print(f"Error during pyslam process_image: {e}")
                # Depending on pyslam's behavior, might need to break or continue
        else:
            # This message will show if actual pyslam initialization is commented out
            # print(f"Timestamp: {current_timestamp:.3f} - Frame not processed (pyslam_system is None).")
            pass # pass can be removed if the print is also removed or commented


        # --- Original Feature Detection and Matching (Commented out for pyslam) ---
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # undistorted_gray_frame = cv2.undistort(gray_frame, camera_matrix, dist_coeffs, None, None)
        # current_keypoints, current_descriptors = detect_features(
        #     undistorted_gray_frame, detector=orb_detector
        # )
        # if prev_descriptors is not None and current_descriptors is not None and len(current_descriptors) > 0 and prev_keypoints is not None:
        #     good_matches = match_features(prev_descriptors, current_descriptors, detector_type='orb')
        #     if good_matches and prev_undistorted_color_frame_for_drawing is not None:
        #         img_matches_display = cv2.drawMatches(...)
        #         cv2.imshow("Feature Matches", img_matches_display)
        #     else:
        #         cv2.imshow("Feature Matches", np.zeros((480, 640, 3), dtype=np.uint8))
        #     if len(good_matches) >= MIN_MATCHES_FOR_POSE:
        #         points1 = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        #         points2 = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        #         R_est, t_est, E_est, mask_pose = estimate_pose(points1, points2, camera_matrix, dist_coeffs=None)
        #         if R_est is not None and t_est is not None:
        #             # ... (Pose accumulation and triangulation logic) ...
        #             # ... (Open3D updates for camera pose and point cloud) ...
        #             pass # Original logic commented out
        #         else: # Pose estimation failed
        #             pass
        #     else: # Not enough good matches
        #         pass
        # else: # No previous descriptors
        #     cv2.imshow("Feature Matches", np.zeros((480, 640, 3), dtype=np.uint8))
        # prev_keypoints = current_keypoints
        # prev_descriptors = current_descriptors
        # prev_undistorted_color_frame_for_drawing = undistorted_color_frame.copy()
        # --- End Original Feature Detection and Matching ---

        # Display the live undistorted feed (can be useful even with pyslam's viewer)
        cv2.imshow("Live Feed to pyslam", undistorted_color_frame)

        # --- Open3D Map Update ---
        if slam_system is not None and 'vis' in locals() and vis.get_window_handle() is not None:
            try:
                # Try to get current camera pose from pyslam
                # This is speculative - the actual method name might be different
                current_pose_matrix = None
                if hasattr(slam_system, 'get_current_pose_matrix'):
                    current_pose_matrix = slam_system.get_current_pose_matrix() # Expected: 4x4 numpy array
                elif hasattr(slam_system, 'get_camera_pose'):
                    current_pose_matrix = slam_system.get_camera_pose() # Expected: 4x4 numpy array

                if current_pose_matrix is not None and isinstance(current_pose_matrix, np.ndarray) and current_pose_matrix.shape == (4, 4):
                    # Remove previous camera pose
                    vis.remove_geometry(current_cam_vis_o3d, reset_bounding_box=False)
                    current_cam_vis_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                    current_cam_vis_o3d.paint_uniform_color([0.1, 0.1, 0.7]) # Blue
                    current_cam_vis_o3d.transform(current_pose_matrix) # Apply the new pose
                    vis.add_geometry(current_cam_vis_o3d, reset_bounding_box=False)

                # Try to get map points from pyslam
                # This is speculative - the actual method name might be different
                map_points = None
                if hasattr(slam_system, 'get_map_points'):
                    map_points = slam_system.get_map_points() # Expected: list or Nx3 numpy array of points
                elif hasattr(slam_system, 'get_point_cloud'):
                    map_points_data = slam_system.get_point_cloud() # Might be an Open3D pointcloud object or raw points
                    if isinstance(map_points_data, o3d.geometry.PointCloud):
                        map_points = np.asarray(map_points_data.points)
                    elif isinstance(map_points_data, np.ndarray):
                        map_points = map_points_data

                if map_points is not None and len(map_points) > 0:
                    if isinstance(map_points, list): # convert to numpy array if list of lists/tuples
                        map_points = np.array(map_points)

                    if isinstance(map_points, np.ndarray) and map_points.ndim == 2 and map_points.shape[1] == 3:
                        point_cloud_o3d.points = o3d.utility.Vector3dVector(map_points)
                        vis.update_geometry(point_cloud_o3d)
                    else:
                        print(f"Debug: map_points received from pyslam is not in expected format (Nx3 numpy array). Shape: {map_points.shape if isinstance(map_points, np.ndarray) else 'Not a numpy array'}")

            except AttributeError as ae:
                print(f"Warning: Pyslam object might not have the expected method for map data: {ae}")
            except Exception as e:
                print(f"Error updating Open3D visualizer: {e}")

        # --- Open3D Visualizer Update (Poll Events) ---
        if 'vis' in locals() and vis.get_window_handle() is not None: # Check if vis exists and window is open
            if not vis.poll_events():
                print("Open3D window was closed by user or events processing failed.")
                break # Exit the main loop
            vis.update_renderer()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User pressed 'q'. Quitting processing loop.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Processing loop finished.")
    
    # --- Pyslam Shutdown (Placeholder) ---
    if slam_system is not None:
        try:
            # NOTE: Speculative call to pyslam's shutdown method.
            # Example:
            slam_system.shutdown()
            print("pyslam system shutdown successfully.") # Modified print
            # pass # Placeholder for actual call # Removed
        except Exception as e:
            print(f"Error during pyslam shutdown: {e}")
    # print("INFO: Pyslam shutdown would be called here if system was initialized.") # Removed
    # --- End Pyslam Shutdown ---

    # --- Open3D Window Destruction ---
    if 'vis' in locals() and vis.get_window_handle() is not None:
        print("Destroying Open3D window...")
        vis.destroy_window()
    print("Application finished.")


if __name__ == "__main__":
    main()
