import cv2
import numpy as np

def estimate_pose(points1, points2, camera_matrix, dist_coeffs=None):
    """
    Estimates the relative pose (Rotation and Translation) between two views
    given corresponding 2D points and camera parameters.

    Args:
        points1 (np.ndarray): NumPy array of 2D points from the first view (shape (N, 2)).
                              These points are assumed to be undistorted.
        points2 (np.ndarray): NumPy array of 2D points from the second view (shape (N, 2)).
                              These points are assumed to be undistorted.
        camera_matrix (np.ndarray): The camera intrinsic matrix.
        dist_coeffs (np.ndarray, optional): Distortion coefficients.
                                           Assumed to be None if points1 and points2 are already undistorted.
                                           Defaults to None.

    Returns:
        tuple: (R, t, E, mask)
               R (np.ndarray): Estimated rotation matrix (3x3).
               t (np.ndarray): Estimated translation vector (3x1).
               E (np.ndarray): Calculated Essential Matrix.
               mask (np.ndarray): Inlier mask from recoverPose.
               Returns (None, None, None, None) if estimation fails.
    """
    if points1 is None or points2 is None:
        print("Error: Input points are None.")
        return None, None, None, None
        
    if not isinstance(points1, np.ndarray) or not isinstance(points2, np.ndarray):
        print("Error: points1 and points2 must be NumPy arrays.")
        return None, None, None, None

    if points1.shape[0] < 5 or points2.shape[0] < 5: # findEssentialMat requires at least 5 points
        print("Error: Need at least 5 point correspondences to estimate Essential Matrix.")
        return None, None, None, None
        
    if points1.shape != points2.shape:
        print("Error: points1 and points2 must have the same shape.")
        return None, None, None, None

    # Ensure points are of type np.float32
    points1_f32 = np.ascontiguousarray(points1, dtype=np.float32)
    points2_f32 = np.ascontiguousarray(points2, dtype=np.float32)

    # Calculate the Essential Matrix
    # Since points are assumed to be undistorted, distCoeffs for findEssentialMat is None.
    # If they were distorted, we would pass 'dist_coeffs' here.
    try:
        # distCoeffs is removed as points1_f32 and points2_f32 are assumed to be already undistorted.
        # Passing it as a keyword argument was causing errors in some OpenCV versions.
        E, mask_e = cv2.findEssentialMat(points1_f32, points2_f32,
                                         camera_matrix,
                                         method=cv2.RANSAC,
                                         prob=0.999,
                                         threshold=1.0)
    except cv2.error as e:
        print(f"OpenCV Error in findEssentialMat: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"Error in findEssentialMat: {e}")
        return None, None, None, None


    if E is None:
        print("Error: Essential Matrix could not be calculated. findEssentialMat returned None.")
        return None, None, None, None

    # Recover Rotation (R) and Translation (t) from the Essential Matrix
    # Again, distCoeffs for recoverPose is None if points1_f32, points2_f32 are undistorted.
    try:
        # distCoeffs is removed as points1_f32 and points2_f32 are assumed to be already undistorted.
        # Removing for consistency with findEssentialMat and to avoid potential keyword argument issues.
        _, R, t, mask_rp = cv2.recoverPose(E, points1_f32, points2_f32,
                                        camera_matrix,
                                        mask=mask_e) # Use the inlier mask from findEssentialMat
    except cv2.error as e:
        print(f"OpenCV Error in recoverPose: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"Error in recoverPose: {e}")
        return None, None, None, None


    if R is None or t is None:
        print("Error: Pose (R or t) could not be recovered. recoverPose returned None.")
        return None, None, None, None

    return R, t, E, mask_rp


if __name__ == '__main__':
    print("--- Testing estimate_pose function ---")

    # Dummy Camera Matrix (approximate for a 640x480 sensor)
    fx, fy = 500, 500
    cx, cy = 320, 240
    dummy_camera_matrix = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0,  1]], dtype=np.float32)
    dummy_dist_coeffs = None # Assuming undistorted points for this test

    # Create some plausible 3D points
    num_points = 20
    true_points_3d = np.hstack([
        (np.random.rand(num_points, 2) - 0.5) * 10, # X, Y between -5 and 5
        (np.random.rand(num_points, 1) + 5)  # Z between 5 and 6
    ])

    # Define a true Rotation and Translation
    true_R_angle_y = np.deg2rad(5) # Small rotation around Y axis
    true_R, _ = cv2.Rodrigues(np.array([0, true_R_angle_y, 0]))
    true_t = np.array([[0.5], [0.1], [-0.2]], dtype=np.float32) # Small translation

    # Project 3D points to 2D for the first view (identity R, t)
    points1_2d, _ = cv2.projectPoints(true_points_3d, 
                                      np.eye(3), np.zeros((3,1)), # R, t for view 1
                                      dummy_camera_matrix, 
                                      dummy_dist_coeffs)
    points1_2d = points1_2d.reshape(-1, 2)

    # Project 3D points to 2D for the second view (using true_R, true_t)
    points2_2d, _ = cv2.projectPoints(true_points_3d,
                                      true_R, true_t, # R, t for view 2
                                      dummy_camera_matrix,
                                      dummy_dist_coeffs)
    points2_2d = points2_2d.reshape(-1, 2)
    
    # Add some noise to 2D points
    noise_level = 0.5 # pixels
    points1_2d += np.random.randn(*points1_2d.shape) * noise_level
    points2_2d += np.random.randn(*points2_2d.shape) * noise_level

    # Introduce some outliers (replace a few points with random ones)
    num_outliers = 3
    if num_points > num_outliers * 2 : # Ensure we have enough points left
        outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
        for i in outlier_indices:
            # Make outlier points significantly different
            points2_2d[i, 0] = np.random.randint(0, cx*2) 
            points2_2d[i, 1] = np.random.randint(0, cy*2)
            # Optionally, also make points1 outliers for some indices
            # points1_2d[i,0] = np.random.randint(0, cx*2)
            # points1_2d[i,1] = np.random.randint(0, cy*2)
        print(f"Introduced {num_outliers} outliers.")


    print(f"Points1 (first 5):\n{points1_2d[:5]}")
    print(f"Points2 (first 5):\n{points2_2d[:5]}")
    
    # Test with insufficient points
    print("\nTesting with insufficient points (should fail gracefully):")
    R_est, t_est, E_est, mask_est = estimate_pose(points1_2d[:3], points2_2d[:3], dummy_camera_matrix, dummy_dist_coeffs)
    if R_est is None:
        print("Insufficient points test passed (returned None).")

    # Test with valid number of points
    print("\nTesting with valid points:")
    R_est, t_est, E_est, mask_est = estimate_pose(points1_2d, points2_2d, dummy_camera_matrix, dummy_dist_coeffs)

    if R_est is not None and t_est is not None:
        print("\nEstimated Rotation Matrix (R_est):\n", R_est)
        print("\nEstimated Translation Vector (t_est):\n", t_est)
        # Note: t_est is a unit vector. Its scale is ambiguous from E matrix alone.
        # The true scale needs to be determined by other means (e.g. triangulation with known baseline).
        
        # Compare with true values (qualitatively, as t_est is unit vector)
        print("\nTrue Rotation Matrix (true_R):\n", true_R)
        print("\nTrue Translation Vector (true_t, normalized for direction comparison):\n", true_t / np.linalg.norm(true_t))
        
        if mask_est is not None:
            num_inliers = np.sum(mask_est)
            print(f"\nNumber of inliers according to recoverPose mask: {num_inliers} out of {len(points1_2d)}")
    else:
        print("\nPose estimation failed with valid points.")
        
    # Test with None inputs
    print("\nTesting with None inputs (should fail gracefully):")
    R_est, t_est, E_est, mask_est = estimate_pose(None, points2_2d, dummy_camera_matrix, dummy_dist_coeffs)
    if R_est is None:
        print("None points1 test passed (returned None).")

    R_est, t_est, E_est, mask_est = estimate_pose(points1_2d, None, dummy_camera_matrix, dummy_dist_coeffs)
    if R_est is None:
        print("None points2 test passed (returned None).")

    # --- Test triangulate_points ---
    if R_est is not None and t_est is not None and mask_est is not None:
        print("\n--- Testing triangulate_points function ---")
        
        # Construct Projection Matrices
        P1 = dummy_camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = dummy_camera_matrix @ np.hstack((R_est, t_est))

        # Get inlier points using the mask from estimate_pose
        # mask_est is (N, 1), convert to boolean array for indexing
        inlier_bool_mask = mask_est.ravel().astype(bool)
        
        inlier_points1_2d = points1_2d[inlier_bool_mask]
        inlier_points2_2d = points2_2d[inlier_bool_mask]
        true_points_3d_inliers = true_points_3d[inlier_bool_mask]

        print(f"Number of inlier points for triangulation: {len(inlier_points1_2d)}")

        if len(inlier_points1_2d) > 0:
            # Option 1: Pass pre-filtered inlier points
            # triangulated_points_3d_option1 = triangulate_points(inlier_points1_2d, inlier_points2_2d, P1, P2)
            
            # Option 2: Pass original points and the mask to test the function's internal masking
            triangulated_points_3d = triangulate_points(points1_2d, points2_2d, P1, P2, inlier_mask=mask_est)


            if triangulated_points_3d is not None:
                print(f"Shape of triangulated 3D points: {triangulated_points_3d.shape}")
                print("First 5 triangulated 3D points (X, Y, Z):\n", triangulated_points_3d[:5])

                # Compare with true 3D points (for inliers)
                # This comparison is tricky due to scale ambiguity of t_est from recoverPose.
                # The triangulated points will be at a scale consistent with t_est (which is unit length).
                # The true_points_3d are at a specific world scale.
                # For a proper MSE, we'd need to align scales (e.g., using Procrustes or by scaling one to match the other).
                
                # Simple approach:
                # 1. Normalize both sets of points (e.g., by translating their centroids to origin and scaling by mean distance)
                # 2. Or, if we know the scale factor between true_t and t_est, apply it.
                #    The scale factor is norm(true_t) / norm(t_est), but norm(t_est) is 1. So scale is norm(true_t).
                
                scale_factor_true_t = np.linalg.norm(true_t)
                # Scale the estimated 3D points by this factor for comparison.
                # This assumes t_est points in roughly the same direction as true_t.
                triangulated_points_3d_scaled = triangulated_points_3d * scale_factor_true_t
                
                # Also, the translation component of true_t needs to be considered if we want a direct MSE
                # on the world coordinates. The triangulated points are relative to camera 1 at origin,
                # and camera 2 at R_est, t_est.
                # If true_points_3d were defined relative to camera 1's CS, then comparison is more direct.
                # Our true_points_3d are in a 'world' frame, and view1 is at origin, view2 is R,t from world.
                # So, triangulated_points_3d_scaled should be comparable to true_points_3d_inliers.

                if triangulated_points_3d_scaled.shape == true_points_3d_inliers.shape:
                    mse = np.mean((triangulated_points_3d_scaled - true_points_3d_inliers)**2)
                    print(f"\nMean Squared Error (MSE) between scaled triangulated points and true inlier 3D points: {mse:.4f}")
                    print("(Note: This MSE depends on the accuracy of R_est, t_est, and the scale factor applied.)")
                else:
                    print("Shape mismatch between triangulated and true points, cannot compute MSE directly.")
            else:
                print("Triangulation failed.")
        else:
            print("No inlier points to triangulate.")
            
    print("\nTesting triangulate_points with empty inputs (should fail gracefully):")
    P1_dummy = dummy_camera_matrix @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2_dummy = dummy_camera_matrix @ np.hstack((np.eye(3), np.ones((3,1)))) # Arbitrary P2
    empty_pts = np.array([])
    res_empty = triangulate_points(empty_pts, empty_pts, P1_dummy, P2_dummy)
    if res_empty is None:
        print("Empty points test passed (returned None).")

    res_empty_mask = triangulate_points(points1_2d, points2_2d, P1_dummy, P2_dummy, inlier_mask=np.zeros_like(mask_est))
    if res_empty_mask is None:
        print("All-zero mask test passed (returned None).")


def triangulate_points(points1, points2, P1, P2, inlier_mask=None):
    """
    Triangulates 3D points from corresponding 2D points in two views.

    Args:
        points1 (np.ndarray): NumPy array of 2D points from the first view (shape (N, 2)).
        points2 (np.ndarray): NumPy array of 2D points from the second view (shape (N, 2)).
        P1 (np.ndarray): The 3x4 projection matrix for the first camera view.
        P2 (np.ndarray): The 3x4 projection matrix for the second camera view.
        inlier_mask (np.ndarray, optional): A boolean or binary mask (shape (N,) or (N, 1))
                                            indicating inlier points. If provided, only
                                            inlier points are triangulated. Defaults to None.

    Returns:
        np.ndarray: NumPy array of triangulated 3D points (shape (N_inliers, 3)),
                    or None if triangulation fails or no inliers.
    """
    if points1 is None or points2 is None or P1 is None or P2 is None:
        print("Error: Input points or projection matrices are None.")
        return None

    if not all(isinstance(arr, np.ndarray) for arr in [points1, points2, P1, P2]):
        print("Error: All inputs (points1, points2, P1, P2) must be NumPy arrays.")
        return None
        
    if points1.shape[0] != points2.shape[0]:
        print("Error: points1 and points2 must have the same number of points (rows).")
        return None

    pts1_to_triangulate = points1.copy()
    pts2_to_triangulate = points2.copy()

    if inlier_mask is not None:
        if not isinstance(inlier_mask, np.ndarray):
            print("Error: inlier_mask must be a NumPy array.")
            return None
        
        # Ensure mask is boolean and 1D for indexing
        if inlier_mask.ndim > 1:
            mask_bool = inlier_mask.ravel().astype(bool)
        else:
            mask_bool = inlier_mask.astype(bool)
            
        if len(mask_bool) != pts1_to_triangulate.shape[0]:
            print(f"Error: Inlier mask shape ({len(mask_bool)}) does not match number of points ({pts1_to_triangulate.shape[0]}).")
            return None
            
        pts1_to_triangulate = pts1_to_triangulate[mask_bool]
        pts2_to_triangulate = pts2_to_triangulate[mask_bool]

    if pts1_to_triangulate.shape[0] == 0:
        print("Warning: No inlier points to triangulate after applying mask (or input was empty).")
        return None

    # Ensure points are float32 or float64 and have shape (2, N) for cv2.triangulatePoints
    # OpenCV expects points to be (2, N) and of type CV_32F or CV_64F.
    # Our input points are (N, 2).
    pts1_for_cv = np.ascontiguousarray(pts1_to_triangulate.T, dtype=np.float32)
    pts2_for_cv = np.ascontiguousarray(pts2_to_triangulate.T, dtype=np.float32)

    try:
        homogeneous_points_4d = cv2.triangulatePoints(P1, P2, pts1_for_cv, pts2_for_cv)
    except cv2.error as e:
        print(f"OpenCV error during triangulation: {e}")
        return None
    except Exception as e:
        print(f"Error during triangulation: {e}")
        return None

    if homogeneous_points_4d is None or homogeneous_points_4d.shape[0] != 4:
        print("Error: Triangulation failed to produce 4D homogeneous points.")
        return None

    # Convert homogeneous coordinates to non-homogeneous (Euclidean)
    # by dividing by the 4th coordinate (w).
    # points_3d will be (3, N_inliers)
    w = homogeneous_points_4d[3, :]
    # Avoid division by zero or very small w; set such points to indicate error or filter them
    # For now, let's proceed, but in a robust system, check w.
    if np.any(np.abs(w) < 1e-6): # Check for very small w values
        print("Warning: Some triangulated points have near-zero w coordinate. Results may be unstable.")
        # Optionally, handle these points (e.g., set to NaN, filter out)
        # For now, we proceed with division.
    
    points_3d_euclidean = homogeneous_points_4d[:3, :] / w

    # Transpose to get shape (N_inliers, 3)
    return points_3d_euclidean.T
