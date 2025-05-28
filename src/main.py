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
import open3d as o3d # New import for Open3D


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

    # --- Open3D Visualization Setup ---
    print("Initializing Open3D visualizer...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Reconstruction")

    point_cloud_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(point_cloud_o3d)

    world_origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(world_origin_axes)

    # Canonical camera axes model (at origin, aligned with XYZ)
    canonical_camera_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    
    # Create the visualizable camera axes object that will be updated
    current_cam_vis_o3d = o3d.geometry.TriangleMesh()
    current_cam_vis_o3d.vertices = canonical_camera_axes.vertices
    current_cam_vis_o3d.triangles = canonical_camera_axes.triangles
    current_cam_vis_o3d.compute_vertex_normals()
    current_cam_vis_o3d.paint_uniform_color([0.1, 0.1, 0.7]) # Example color: dark blue
    vis.add_geometry(current_cam_vis_o3d)

    # Optional: Initial view control settings
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    view_control.set_lookat([0, 0, 0]) # Look at the origin
    view_control.set_up([0, -1, 0])    # OpenCV's Y is down, Open3D's Y is up. This aligns them.
    view_control.set_front([0, 0, -1]) # Look along the negative Z axis

    # Accumulated pose of the 'previous' camera in the world (starts at origin)
    world_R_previous = np.eye(3, dtype=np.float64)
    world_t_previous = np.zeros((3, 1), dtype=np.float64)

    # Pose of the 'current' camera in the world, to be displayed
    display_R = np.eye(3, dtype=np.float64)
    display_t = np.zeros((3, 1), dtype=np.float64)

    print("Open3D visualizer initialized.")

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
                    
                    # Store the pose of the previous camera (N-1) in the world, used for transforming points_3d
                    prev_cam_R_world = world_R_previous.copy()
                    prev_cam_t_world = world_t_previous.copy()
                    
                    # Accumulate pose: T_world_current = T_world_previous * T_previous_current
                    # R_est, t_est is T_previous_current (pose of current cam N relative to previous cam N-1)
                    # display_R, display_t will be T_world_current (pose of current cam N in world)
                    display_R = prev_cam_R_world @ R_est
                    display_t = prev_cam_t_world + prev_cam_R_world @ t_est

                    # Update Open3D camera_axes_o3d pose using display_R, display_t
                    if display_R is not None and display_t is not None:
                        transform_matrix = np.eye(4)
                        transform_matrix[:3, :3] = display_R
                        transform_matrix[:3, 3] = display_t.squeeze()

                        # Start with a fresh copy of the canonical axes
                        temp_cam_axes = o3d.geometry.TriangleMesh()
                        temp_cam_axes.vertices = canonical_camera_axes.vertices
                        temp_cam_axes.triangles = canonical_camera_axes.triangles
                        
                        # Apply the transformation
                        temp_cam_axes.transform(transform_matrix)
                        
                        # Update the vertices and triangles of the object already in the scene
                        current_cam_vis_o3d.vertices = temp_cam_axes.vertices
                        current_cam_vis_o3d.triangles = temp_cam_axes.triangles
                        current_cam_vis_o3d.compute_vertex_normals()
                        # current_cam_vis_o3d.paint_uniform_color([0.1, 0.1, 0.7]) # Already painted at init
                        
                        vis.update_geometry(current_cam_vis_o3d)
                    
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
                        
                        # Update Open3D point_cloud_o3d with points_3d_world
                        o3d_points = o3d.utility.Vector3dVector(points_3d_world)
                        point_cloud_o3d.points = o3d_points
                        # Optional: Add colors (e.g., uniform gray if no other color info)
                        # if not point_cloud_o3d.has_colors():
                        #    point_cloud_o3d.paint_uniform_color([0.7, 0.7, 0.7])
                        vis.update_geometry(point_cloud_o3d)

                    else:
                        print("Triangulation failed or yielded no 3D points.")
                        # Keep last known point cloud, or clear it:
                        # point_cloud_o3d.clear()
                        # vis.update_geometry(point_cloud_o3d)
                        pass # Explicitly doing nothing to keep last cloud

                    # Update world_R_previous and world_t_previous for the *next* iteration
                    # They become the pose of the current camera (N) in the world
                    world_R_previous = display_R.copy()
                    world_t_previous = display_t.copy()

                else: # R_est is None or t_est is None (pose estimation failed)
                    print("Pose estimation failed for this frame pair.")
                    # Do not update world_R_previous, world_t_previous here, keep last good pose.
                    # Also, display_R, display_t are not updated, so current_cam_vis_o3d in Open3D remains at last good pose.
                    # If pose estimation failed, we might not have new points or they might be unreliable.
                    # Consider clearing point cloud or leaving it. For now, it's handled by points_3d_relative_to_prev_cam check.
                    pass 
            else:
                print(f"Not enough good matches for pose estimation (found {len(good_matches)}, need {MIN_MATCHES_FOR_POSE}).")
                # No new points, so decision to clear or keep existing cloud applies.
                # For now, leave existing points.
                pass
        else:
            # Clear the matches window if no previous descriptors or current descriptors
            cv2.imshow("Feature Matches", np.zeros((480, 640, 3), dtype=np.uint8))
            # No new points, so decision to clear or keep existing cloud applies.
            # For now, leave existing points.
            pass

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

        # Update Open3D visualizer
        if not vis.poll_events(): # Process window events and check if closed
            print("Open3D window was closed by user or events processing failed.")
            break                 # Exit loop if window was closed
        vis.update_renderer()     # Render the updated scene
        
        # Process OpenCV window events and check for 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User pressed 'q'. Quitting feature tracking loop.")
            break


    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Feature tracking loop finished.")
    
    if vis:
        print("Destroying Open3D window...")
        vis.destroy_window()
    print("Application finished.")


if __name__ == "__main__":
    main()
