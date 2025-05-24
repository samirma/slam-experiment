import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Main SfM System Function ---
def run_sfm_system():
    # --- Phase 1: Initialization ---
    # 1. Imports are implicitly handled by being at the top of the module.

    # 2. Load camera calibration parameters
try:
    calibration_data = np.load("calibration_params.npz")
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']
    print("Calibration parameters loaded successfully.")
except FileNotFoundError:
    print("Error: calibration_params.npz not found. Please run calibrate_camera.py first.")
    exit()

# 3. Initialize ORB detector and Brute-Force Matcher
orb = cv2.ORB_create(nfeatures=2000) # Increased features for more robust tracking
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 4. Initialize global SfM variables
world_points_3D = np.empty((0, 3), dtype=np.float32)
# ORB descriptors are 32 bytes long (uint8)
world_points_des = np.empty((0, 32), dtype=np.uint8) 
camera_poses = [] # List of dictionaries, each with 'R' and 't'

# 5. Variables for first two frames
frame_count = 0
first_frame_data = None # Will store {'kps': ..., 'des': ..., 'img': ...}
kps_prev = None
des_prev = None
prev_frame_color = None

# 6. Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting Incremental SfM: Initialization Phase")
print("Look at the scene and press 'c' to capture frames for initialization.")

# 7. Loop for initialization
init_success = False
while frame_count < 2:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break
    
    current_frame_color_undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
    gray_curr = cv2.cvtColor(current_frame_color_undistorted, cv2.COLOR_BGR2GRAY)
    kps_curr, des_curr = orb.detectAndCompute(gray_curr, None)

    display_frame = current_frame_color_undistorted.copy()
    if kps_curr is not None:
        cv2.drawKeypoints(display_frame, kps_curr, display_frame, color=(0,255,0))

    cv2.putText(display_frame, f"Capturing frame {frame_count+1}/2. Press 'c'.", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_frame, "Press 'q' to quit.", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    cv2.imshow("Incremental SfM - Initialization", display_frame)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quitting during initialization.")
        cap.release()
        cv2.destroyAllWindows()
        exit()
    
    if key == ord('c'):
        if des_curr is None or len(des_curr) < 10: # Need enough features
            print("Not enough features detected in the current frame. Try again.")
            continue

        frame_count += 1
        print(f"Frame {frame_count}/2 captured.")

        if frame_count == 1:
            first_frame_data = {
                'kps': kps_curr, 
                'des': des_curr, 
                'img': current_frame_color_undistorted.copy()
            }
        elif frame_count == 2:
            kps1, des1 = first_frame_data['kps'], first_frame_data['des']
            kps2, des2 = kps_curr, des_curr # Current frame is the second frame

            matches = bf.match(des1, des2)
            # Sort by distance for good matches
            good_matches = sorted(matches, key=lambda x: x.distance)
            # Filter good matches - e.g. top 100 or distance based
            good_matches = [m for m in good_matches if m.distance < 50][:150] # Heuristic distance threshold

            if len(good_matches) < 20: # Need a reasonable number of matches
                print(f"Not enough good matches ({len(good_matches)}) for initialization. Resetting.")
                frame_count = 0
                first_frame_data = None
                continue

            pts1 = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            E, mask_e = cv2.findEssentialMat(pts1, pts2, mtx, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None:
                print("Could not estimate Essential Matrix. Resetting.")
                frame_count = 0
                first_frame_data = None
                continue
            
            print("Essential Matrix Estimated.")
            retval, R_init, t_init, mask_rp = cv2.recoverPose(E, pts1, pts2, mtx, mask=mask_e)

            if not retval or R_init is None or t_init is None or mask_rp is None or np.sum(mask_rp) < 10 :
                print("Could not recover pose or not enough inliers. Resetting.")
                frame_count = 0
                first_frame_data = None
                continue
            
            print(f"Pose Recovered. Inliers: {np.sum(mask_rp > 0)}")

            # Initial poses
            R_w0, t_w0 = np.eye(3), np.zeros((3, 1), dtype=np.float32)
            R_w1, t_w1 = R_init, t_init.astype(np.float32) # Ensure t_w1 is float32

            camera_poses.append({'R': R_w0, 't': t_w0})
            camera_poses.append({'R': R_w1, 't': t_w1})

            # Projection matrices
            P1 = mtx @ np.hstack((R_w0, t_w0))
            P2 = mtx @ np.hstack((R_w1, t_w1))
            
            # Filter points using recoverPose mask
            pts1_inliers = pts1[mask_rp.ravel() > 0]
            pts2_inliers = pts2[mask_rp.ravel() > 0]
            
            points4D_hom = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T) # Ensure (2,N) shape
            current_3d_points = (points4D_hom[:3, :] / points4D_hom[3, :]).T # Shape (N, 3)
            
            # Descriptor selection for world_points_des
            # Select descriptors from the first frame (des1) that correspond to inlier matches
            des_indices_inliers1 = [m.queryIdx for i, m in enumerate(good_matches) if mask_rp[i] > 0]
            des_for_world_points = des1[des_indices_inliers1]

            world_points_3D = current_3d_points.astype(np.float32)
            world_points_des = des_for_world_points.astype(np.uint8)

            kps_prev = kps2
            des_prev = des2
            prev_frame_color = current_frame_color_undistorted.copy()
            
            print("Initialization complete. Starting tracking phase.")
            print(f"Initial 3D points: {len(world_points_3D)}")
            init_success = True
            break # Exit initialization loop

if not init_success:
    print("Initialization failed. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# --- Phase 2: Tracking and Incremental SfM (Main Loop) ---
print("\nStarting Tracking Phase...")
cv2.destroyWindow("Incremental SfM - Initialization") # Close init window

frame_idx_global = 2 # We've processed 2 frames already
plot_update_counter = 0
plot_update_freq = 5 # Update plot every 5 frames, adjust as needed

# Initialize Matplotlib for real-time plotting
plt.ion() 
fig_3d = plt.figure(figsize=(12, 8)) # Adjusted size
ax_3d = fig_3d.add_subplot(111, projection='3d')


# --- Plotting Function ---
def plot_3d_scene(ax, camera_poses_list, points_3d_world, title="3D Scene"):
    ax.clear() # Clear previous frame

    # Plot 3D points
    if points_3d_world is not None and len(points_3d_world) > 0:
        ax.scatter(points_3d_world[:, 0], points_3d_world[:, 1], points_3d_world[:, 2], c='b', marker='.', s=5, label='3D Points')

    # Plot camera poses
    cam_centers = []
    if camera_poses_list is not None:
        for i, pose in enumerate(camera_poses_list):
            R_wc = pose['R'] # World to Cam
            t_wc = pose['t'] # World to Cam
            
            # Camera center in world coords: C = -R_wc.T @ t_wc
            cam_center_world = -R_wc.T @ t_wc
            cam_centers.append(cam_center_world.ravel())
            
            # Plot camera orientation (e.g., Z-axis of camera in world)
            # Z-axis in camera coords is (0,0,1). In world coords: R_wc.T @ (0,0,1).T = R_wc.T[:,2]
            z_axis_cam_world = R_wc.T[:, 2]
            ax.plot([cam_center_world[0], cam_center_world[0] + z_axis_cam_world[0]*0.5], # Scale axis for visibility
                    [cam_center_world[1], cam_center_world[1] + z_axis_cam_world[1]*0.5],
                    [cam_center_world[2], cam_center_world[2] + z_axis_cam_world[2]*0.5], 
                    color='r' if i == 0 else ('m' if i == len(camera_poses_list)-1 else 'g')) # First red, last magenta, others green

    if cam_centers:
        cam_centers_np = np.array(cam_centers)
        ax.scatter(cam_centers_np[:, 0], cam_centers_np[:, 1], cam_centers_np[:, 2], c='k', marker='^', s=30, label='Camera Poses')
        if len(cam_centers_np) > 1:
            ax.plot(cam_centers_np[:, 0], cam_centers_np[:, 1], cam_centers_np[:, 2], color='k', linestyle='-', linewidth=1)

    ax.set_xlabel('X world')
    ax.set_ylabel('Y world')
    ax.set_zlabel('Z world')
    ax.set_title(title)
    
    # Auto-scaling axes logic (simplified for real-time, can be improved)
    all_plot_points = []
    if points_3d_world is not None and len(points_3d_world) > 0:
        all_plot_points.append(points_3d_world)
    if cam_centers:
        all_plot_points.append(np.array(cam_centers))
    
    if all_plot_points:
        combined_points = np.vstack(all_plot_points)
        if combined_points.shape[0] > 0:
            max_range = np.array([combined_points[:,0].max()-combined_points[:,0].min(), 
                                  combined_points[:,1].max()-combined_points[:,1].min(), 
                                  combined_points[:,2].max()-combined_points[:,2].min()]).max() / 2.0
            if max_range == 0: max_range = 5 # Default if no range (e.g. single point)

            mid_x = (combined_points[:,0].max()+combined_points[:,0].min()) * 0.5
            mid_y = (combined_points[:,1].max()+combined_points[:,1].min()) * 0.5
            mid_z = (combined_points[:,2].max()+combined_points[:,2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        else: # Default view if no points
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(0, 10)
    else: # Default view if no points or cameras
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(0, 10)

    ax.legend(fontsize='small')


while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break
    
    current_frame_color = cv2.undistort(frame, mtx, dist, None, mtx)
    gray_curr = cv2.cvtColor(current_frame_color, cv2.COLOR_BGR2GRAY)
    kps_curr, des_curr = orb.detectAndCompute(gray_curr, None)

    display_tracking_frame = current_frame_color.copy()

    if des_curr is None or len(des_curr) < 5: # Need some descriptors to proceed
        print("Not enough features in current frame, skipping PnP and Triangulation.")
        kps_prev = kps_curr
        des_prev = des_curr
        prev_frame_color = current_frame_color.copy()
        cv2.imshow("Incremental SfM - Tracking", display_tracking_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    # c. Pose Estimation (PnP)
    R_curr_world_to_cam, t_curr_world_to_cam = None, None # Initialize for this frame
    if len(world_points_3D) > 5 and len(world_points_des) > 5 and len(world_points_des) == len(world_points_3D):
        pnp_matches = bf.match(des_curr, world_points_des) # query: current, train: world
        pnp_matches = sorted(pnp_matches, key=lambda x: x.distance)
        
        # Filter matches for PnP (e.g., top N or distance-based)
        good_pnp_matches = [m for m in pnp_matches if m.distance < 60][:100] # Heuristic

        if len(good_pnp_matches) >= 5: # Need at least 5 points for PnP
            image_pts_pnp = np.float32([kps_curr[m.queryIdx].pt for m in good_pnp_matches]).reshape(-1,1,2)
            object_pts_pnp = np.float32([world_points_3D[m.trainIdx] for m in good_pnp_matches]).reshape(-1,1,3)
            
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(object_pts_pnp, image_pts_pnp, mtx, dist, iterationsCount=100, reprojectionError=8.0)
                if success and inliers is not None and len(inliers) >= 5:
                    R_cam_to_world, _ = cv2.Rodrigues(rvec)
                    # Convert to World-to-Camera pose for projection matrix
                    R_curr_world_to_cam = R_cam_to_world.T
                    t_curr_world_to_cam = -R_cam_to_world.T @ tvec.astype(np.float32) # Ensure tvec is float32

                    camera_poses.append({'R': R_curr_world_to_cam, 't': t_curr_world_to_cam})
                    print(f"Frame {frame_idx_global}: Pose Estimated. Total poses: {len(camera_poses)}. PnP Inliers: {len(inliers)}/{len(good_pnp_matches)}")

                    # Draw PnP inliers
                    for idx in inliers.ravel():
                        pt_2d = tuple(map(int, image_pts_pnp[idx].ravel()))
                        cv2.circle(display_tracking_frame, pt_2d, 5, (255, 0, 0), -1) # Blue circles for PnP inliers
                else:
                    print(f"Frame {frame_idx_global}: PnP failed or not enough inliers. Success: {success}, Inliers: {len(inliers) if inliers is not None else 'None'}")
            except cv2.error as e:
                print(f"Frame {frame_idx_global}: cv2.solvePnPRansac error: {e}")

        else:
            print(f"Frame {frame_idx_global}: Not enough good matches for PnP ({len(good_pnp_matches)}).")
    else:
        print(f"Frame {frame_idx_global}: Not enough 3D points or descriptors in world map for PnP.")


    # d. Triangulate New Points
    # Only triangulate if pose was successfully estimated in this frame OR if this is the first frame after init (where prev pose is known)
    if (R_curr_world_to_cam is not None and t_curr_world_to_cam is not None) and \
       (des_prev is not None and len(des_prev) > 0):
        
        new_point_matches = bf.match(des_curr, des_prev) # query: current, train: previous
        new_point_matches = sorted(new_point_matches, key=lambda x: x.distance)
        good_new_matches = [m for m in new_point_matches if m.distance < 50][:100] # Heuristic

        if len(good_new_matches) > 10: # Need enough matches to triangulate
            pts_curr_tri = np.float32([kps_curr[m.queryIdx].pt for m in good_new_matches]).reshape(-1, 1, 2)
            pts_prev_tri = np.float32([kps_prev[m.trainIdx].pt for m in good_new_matches]).reshape(-1, 1, 2)

            # Get last two poses for triangulation
            # Current pose is the one just estimated by PnP (R_curr_world_to_cam, t_curr_world_to_cam)
            # Previous pose is the last one in camera_poses list BEFORE adding the current one
            if len(camera_poses) >= 1: # Current pose already added if PnP was successful
                pose_curr_dict = camera_poses[-1]
                # Previous pose: if PnP was successful, it's camera_poses[-2]. If PnP failed, we might not have a current pose.
                # This logic assumes PnP gave us the current pose. If PnP failed, we might skip triangulation or use an older pose.
                # For simplicity, if PnP succeeded, we use the last two.
                if len(camera_poses) >= 2:
                    pose_prev_dict = camera_poses[-2] 
                else: # This means PnP just gave the first pose after initialization
                    pose_prev_dict = camera_poses[0] # Should be the second pose from init (R_w1, t_w1)

                P_curr = mtx @ np.hstack((pose_curr_dict['R'], pose_curr_dict['t']))
                P_prev = mtx @ np.hstack((pose_prev_dict['R'], pose_prev_dict['t']))

                # Triangulate
                new_points_4D_hom = cv2.triangulatePoints(P_prev, P_curr, pts_prev_tri.reshape(-1,2).T, pts_curr_tri.reshape(-1,2).T)
                new_points_3D_cand = (new_points_4D_hom[:3, :] / new_points_4D_hom[3, :]).T # (N,3)

                # Basic cheirality check (points must be in front of both cameras)
                # For P_curr: Rz * (X - Cz) > 0.  X is point in world. Cam center Cz = -R.T @ t
                # Transform points to camera coordinates: X_cam = R @ X + t
                valid_pts_mask = np.ones(len(new_points_3D_cand), dtype=bool)
                
                # Check for current camera
                pts_curr_cam = (pose_curr_dict['R'] @ new_points_3D_cand.T) + pose_curr_dict['t']
                valid_pts_mask &= (pts_curr_cam[2,:] > 0) # Z > 0 in camera frame
                
                # Check for previous camera
                pts_prev_cam = (pose_prev_dict['R'] @ new_points_3D_cand.T) + pose_prev_dict['t']
                valid_pts_mask &= (pts_prev_cam[2,:] > 0)

                new_points_3D = new_points_3D_cand[valid_pts_mask]
                
                if len(new_points_3D) > 0:
                    # Get descriptors for these new valid 3D points (from current frame)
                    des_indices_new_pts = [m.queryIdx for i, m in enumerate(good_new_matches) if valid_pts_mask[i]]
                    new_des_for_3D = des_curr[des_indices_new_pts]

                    # TODO: Add filtering to avoid adding points too close to existing world_points_3D
                    # or whose descriptors are too similar to existing world_points_des

                    world_points_3D = np.vstack((world_points_3D, new_points_3D.astype(np.float32)))
                    world_points_des = np.vstack((world_points_des, new_des_for_3D.astype(np.uint8)))
                    print(f"Frame {frame_idx_global}: Triangulated {len(new_points_3D)} new points. Total map points: {len(world_points_3D)}")
                else:
                    print(f"Frame {frame_idx_global}: No new points passed cheirality check.")
            else:
                 print(f"Frame {frame_idx_global}: Not enough camera poses for triangulation.")
        else:
            print(f"Frame {frame_idx_global}: Not enough good matches for new point triangulation ({len(good_new_matches)}).")


    # e. Update previous frame data
    kps_prev = kps_curr
    des_prev = des_curr
    prev_frame_color = current_frame_color.copy()
    frame_idx_global += 1

    # f. Display frame
    cv2.putText(display_tracking_frame, f"Frame: {frame_idx_global}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
    cv2.putText(display_tracking_frame, "Press 'q' to quit.", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    cv2.imshow("Incremental SfM - Tracking", display_tracking_frame)

    # e. Update previous frame data
    kps_prev = kps_curr
    des_prev = des_curr
    prev_frame_color = current_frame_color.copy()
    frame_idx_global += 1

    # f. Display frame
    cv2.putText(display_tracking_frame, f"Frame: {frame_idx_global}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
    cv2.putText(display_tracking_frame, "Press 'q' to quit.", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    cv2.imshow("Incremental SfM - Tracking", display_tracking_frame)

    # Update 3D plot
    if frame_idx_global % plot_update_freq == 0 and len(world_points_3D) > 0:
        plot_3d_scene(ax_3d, camera_poses, world_points_3D, 
                      title=f"SfM: {len(world_points_3D)} pts, {len(camera_poses)} poses (Frame {frame_idx_global})")
        plt.pause(0.01) # Crucial for allowing redraw

    # TODO: Consider triggering Bundle Adjustment here periodically
    # if frame_idx_global % BA_INTERVAL == 0: # BA_INTERVAL could be e.g. 5 or 10 frames
    #     # Need to prepare point_observations structure. This is complex.
    #     # It requires tracking which 3D point is observed by which camera and its 2D coordinates.
    #     # For example, observations = [(cam_idx, point_3d_idx, x, y), ...]
    #     # print(f"Frame {frame_idx_global}: Triggering periodic Bundle Adjustment (placeholder).")
    #     # temp_observations_placeholder = [] # Actual observations would be collected
    #     # camera_poses, world_points_3D = run_bundle_adjustment(camera_poses, world_points_3D, temp_observations_placeholder, mtx)
    #     pass


    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting tracking phase.")
        break

# 10. Release capture, destroy windows
cap.release()
cv2.destroyAllWindows()
plt.ioff() # Turn off interactive mode
if fig_3d: # Ensure fig_3d exists
    plt.close(fig_3d) # Close the Matplotlib figure
print("Processing finished.")


# --- Bundle Adjustment Placeholder Function ---
def run_bundle_adjustment(camera_params_list, points_3d, point_observations, camera_intrinsics, n_iter=10):
    # Placeholder for Bundle Adjustment
    # This function would refine camera_params_list (poses) and points_3d
    #
    # Args:
    #   camera_params_list: List of dictionaries, each with 'R' (rotation matrix) and 't' (translation vector)
    #   points_3d: Nx3 numpy array of 3D world points
    #   point_observations: A list of tuples, where each tuple is (camera_idx, point_idx, observed_x, observed_y)
    #                       camera_idx refers to the index in camera_params_list
    #                       point_idx refers to the index in points_3d
    #   camera_intrinsics: The camera matrix (mtx)
    #   n_iter: Number of iterations for the optimization (if using scipy.optimize.least_squares)
    #
    # Returns:
    #   refined_camera_params_list, refined_points_3d
    
    print("\n--- Bundle Adjustment (Placeholder) ---")
    print(f"Would optimize {len(camera_params_list)} camera poses and {len(points_3d)} 3D points.")
    print(f"Based on {len(point_observations)} observations.")
    
    # Example of what BA does (conceptual):
    # 1. Construct a big non-linear least squares problem.
    #    Variables: All elements of R and t for each camera, and all x,y,z for each point.
    #    Residuals: For each observation, calculate reprojection_error = observed_xy - project(Point_3D_world, CameraPose, Intrinsics)
    # 2. Solve using an iterative solver (e.g., Levenberg-Marquardt).
    #    scipy.optimize.least_squares could be a starting point for a simpler BA.
    
    # For now, just return the inputs as BA is not implemented
    return camera_params_list, points_3d


# --- Final Bundle Adjustment (Placeholder Call) ---
# This section can remain if a final BA run is desired after the loop, 
# but the visualization itself is now handled dynamically.
all_observations_placeholder = [] 

if len(camera_poses) > 0 and len(world_points_3D) > 0:
    print("\nCalling Final Bundle Adjustment (Placeholder)...")
    # Note: BA might refine camera_poses and world_points_3D.
    # If the plot is still open, it won't auto-update with these changes unless plot_3d_scene is called again.
    camera_poses, world_points_3D = run_bundle_adjustment(camera_poses, world_points_3D, all_observations_placeholder, mtx)
    print("Final Bundle Adjustment (Placeholder) finished.")

    # Optionally, a final static plot if desired after BA and loop exit:
    # plt.ioff() # Ensure interactive mode is off if it was on
    # final_fig = plt.figure(figsize=(12, 8))
    # final_ax = final_fig.add_subplot(111, projection='3d')
    # plot_3d_scene(final_ax, camera_poses, world_points_3D, title="Final SfM Result (After BA)")
    # plt.show() # This will block until closed


# The old static plot (11.) is removed as dynamic plotting is implemented.
# If you want a final plot to inspect *after* the loop, you can re-enable a static plot call here,
# ensuring plt.ioff() is called before it if plt.ion() was used.

print("Script finished.")
    # plt.show() # This would be needed if you want to keep the last dynamic plot open and interactive after loop exit.
    # However, since we call plt.close(fig_3d), it will be gone.
    # If you want to inspect the very last state, consider a final static plot as commented above,
    # or remove the plt.close(fig_3d) and ensure plt.ioff() is handled.

if __name__ == "__main__":
    print("Running Incremental SfM directly.")
    # A simple check for calibration file, similar to main.py
    try:
        with open("calibration_params.npz", "rb") as f:
            pass # File exists
        print("'calibration_params.npz' found.")
    except FileNotFoundError:
        print("CRITICAL ERROR: 'calibration_params.npz' not found.")
        print("Please run 'calibrate_camera.py' first to generate this file.")
        exit()
    
    run_sfm_system()
    print("SfM system shut down (run directly).")
