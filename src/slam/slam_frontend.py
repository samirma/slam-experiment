import cv2
import numpy as np
# This will be: from src.utils import CameraParams
# For now, to allow development if utils is not in the exact expected relative path during dev:
try:
    from src.utils import CameraParams
except ImportError:
    # Fallback for cases where the script might be run directly or path issues
    # This assumes camera_params.py is discoverable in PYTHONPATH or similar context
    # In a structured project, the above relative import is preferred.
    print("Attempting fallback import for CameraParams. Ensure PYTHONPATH is set for src.")
    from utils.camera_params import CameraParams


class SLAMFrontend:
    """
    Implements the frontend of a Monocular SLAM system.
    Handles feature extraction, matching, pose estimation, and initial map creation.
    """
    def __init__(self, camera_params: CameraParams):
        """
        Initializes the SLAM Frontend.

        Args:
            camera_params (CameraParams): Camera intrinsic parameters.
        """
        self.camera_params: CameraParams = camera_params
        self.K: np.ndarray = self.camera_params.get_K()
        self.dist_coeffs: np.ndarray = self.camera_params.get_dist_coeffs()

        # Feature detection and matching
        self.orb: cv2.ORB = cv2.ORB_create(nfeatures=1500,
                                           scaleFactor=1.2,
                                           nlevels=8,
                                           edgeThreshold=31,
                                           firstLevel=0,
                                           WTA_K=2,
                                           scoreType=cv2.ORB_HARRIS_SCORE,
                                           patchSize=31,
                                           fastThreshold=20)
        self.matcher: cv2.BFMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # SLAM state variables
        self.current_R_w2c: np.ndarray = np.eye(3, dtype=np.float32) # Pose of world in current camera frame
        self.current_t_w2c: np.ndarray = np.zeros((3, 1), dtype=np.float32)

        self.prev_kps: list[cv2.KeyPoint] | None = None # Keypoints from the strictly previous frame
        self.prev_des: np.ndarray | None = None       # Descriptors from the strictly previous frame
        
        self.kf1_data: dict | None = None # Data for the first keyframe {'gray': img, 'kps': kps, 'des': des}
        self.keyframes: list[dict] = []   # List to store keyframes (pose, kps, des, map_point_descriptors)
        self.map_points: np.ndarray = np.array([], dtype=np.float32).reshape(0, 3) # 3D map points (Nx3)
        self.map_initial_descriptors: np.ndarray | None = None # Descriptors corresponding to self.map_points

        self.state: str = "WAITING_FOR_FIRST_FRAME"
        self.min_features_for_tracking: int = 20 # Min good matches for pose recovery / triangulation
        self.min_features_for_pnp: int = 5      # Min good matches for solvePnP
        self.initial_baseline_scale: float = 0.1 # Desired scale for the baseline translation between KF1 and KF2
        
        # Keyframe selection criteria
        self.min_kf_translation_dist: float = 0.2 # meters (increased from 0.1 for more spacing)
        self.min_kf_rotation_angle: float = 10.0  # degrees (increased for more spacing)


    def _extract_features(self, frame_gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        kps, des = self.orb.detectAndCompute(frame_gray, None)
        return kps, des

    def _match_features(self, des_prev: np.ndarray, des_curr: np.ndarray) -> list[cv2.DMatch]:
        if des_prev is None or des_curr is None or des_prev.shape[0] == 0 or des_curr.shape[0] == 0:
            return []
        if des_prev.dtype != np.uint8:
            des_prev = des_prev.astype(np.uint8)
        if des_curr.dtype != np.uint8:
            des_curr = des_curr.astype(np.uint8)
        matches = self.matcher.match(des_prev, des_curr)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def _calculate_camera_movement(self, R1, t1, R2, t2) -> tuple[float, float]:
        """Calculates translation distance and rotation angle between two poses."""
        translation_dist = np.linalg.norm(t2 - t1)
        
        # Relative rotation: R_rel = R2 @ R1.T
        R_rel = R2 @ R1.T
        # Convert rotation matrix to axis-angle
        angle_rad, _ = cv2.Rodrigues(R_rel)
        # Angle in degrees
        rotation_angle_deg = np.rad2deg(np.linalg.norm(angle_rad))
        
        return translation_dist, rotation_angle_deg

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[cv2.KeyPoint], list[cv2.DMatch]]:
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kps, des = self._extract_features(frame_gray)
        
        if kps is None: kps = []
        if des is None: des = np.array([], dtype=np.uint8).reshape(0, self.orb.descriptorSize())

        viz_kps = kps 
        viz_matches: list[cv2.DMatch] = []

        if self.state == "WAITING_FOR_FIRST_FRAME":
            if len(kps) >= self.min_features_for_tracking:
                self.kf1_data = {'gray': frame_gray, 'kps': kps, 'des': des}
                kf1_pose_R = np.eye(3, dtype=np.float32)
                kf1_pose_t = np.zeros((3, 1), dtype=np.float32)
                self.keyframes.append({'R': kf1_pose_R, 't': kf1_pose_t, 'kps': kps, 'des': des})
                self.current_R_w2c = kf1_pose_R
                self.current_t_w2c = kf1_pose_t
                self.state = "WAITING_FOR_SECOND_FRAME"
            self.prev_kps, self.prev_des = kps, des
            return self.current_R_w2c, self.current_t_w2c, viz_kps, viz_matches

        elif self.state == "WAITING_FOR_SECOND_FRAME":
            if self.kf1_data is None or self.kf1_data['des'] is None:
                self.prev_kps, self.prev_des = kps, des
                return self.current_R_w2c, self.current_t_w2c, viz_kps, viz_matches

            matches = self._match_features(self.kf1_data['des'], des)
            viz_matches = matches

            if len(matches) >= self.min_features_for_tracking:
                pts_kf1_matched_all = np.float32([self.kf1_data['kps'][m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                pts_curr_matched_all = np.float32([kps[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                
                # Store descriptors of current frame's matched points for potential map_initial_descriptors
                des_curr_matched_all = np.array([des[m.trainIdx] for m in matches])


                pts_kf1_undistorted = cv2.undistortPoints(pts_kf1_matched_all, self.K, self.dist_coeffs, P=self.K)
                pts_curr_undistorted = cv2.undistortPoints(pts_curr_matched_all, self.K, self.dist_coeffs, P=self.K)

                if pts_kf1_undistorted is None or pts_curr_undistorted is None:
                    self.prev_kps, self.prev_des = kps, des
                    return self.current_R_w2c, self.current_t_w2c, viz_kps, viz_matches
                
                E, e_mask = cv2.findEssentialMat(pts_curr_undistorted, pts_kf1_undistorted,
                                                 self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

                if E is not None and e_mask is not None:
                    e_mask_bool = e_mask.ravel() == 1
                    pts_kf1_undistorted_inliers = pts_kf1_undistorted[e_mask_bool]
                    pts_curr_undistorted_inliers = pts_curr_undistorted[e_mask_bool]
                    
                    # Get the descriptors for the inliers in the current (second) frame
                    des_curr_inliers = des_curr_matched_all[e_mask_bool]

                    if len(pts_curr_undistorted_inliers) >= self.min_features_for_tracking:
                        num_inliers, R_rel, t_rel, _ = cv2.recoverPose(
                            E, pts_curr_undistorted_inliers, pts_kf1_undistorted_inliers, self.K)

                        if R_rel is not None and t_rel is not None and num_inliers >= self.min_features_for_tracking:
                            current_t_norm = np.linalg.norm(t_rel)
                            scaled_t_rel = t_rel * (self.initial_baseline_scale / current_t_norm) if current_t_norm > 1e-5 else t_rel
                            
                            R_kf2 = R_rel 
                            t_kf2 = scaled_t_rel
                            self.keyframes.append({'R': R_kf2, 't': t_kf2, 'kps': kps, 'des': des}) # Store all kps/des for KF2
                            self.current_R_w2c, self.current_t_w2c = R_kf2, t_kf2

                            P1 = self.K @ np.hstack((np.eye(3), np.zeros((3,1)))) 
                            P2 = self.K @ np.hstack((R_kf2, t_kf2))
                            
                            pts_kf1_inliers_rs = pts_kf1_undistorted_inliers.reshape(-1, 2).T
                            pts_curr_inliers_rs = pts_curr_undistorted_inliers.reshape(-1, 2).T
                            
                            homogeneous_points = cv2.triangulatePoints(P1, P2, pts_kf1_inliers_rs, pts_curr_inliers_rs)
                            if homogeneous_points is not None and homogeneous_points.shape[1] > 0:
                                self.map_points = (homogeneous_points[:3,:] / homogeneous_points[3,:]).T 
                                self.map_initial_descriptors = des_curr_inliers # Store descriptors of KF2 inliers for PnP
                            
                            self.state = "TRACKING_WITH_LOCAL_MAP" # <<<< STATE RENAME
            
            self.prev_kps, self.prev_des = kps, des
            return self.current_R_w2c, self.current_t_w2c, viz_kps, viz_matches

        elif self.state == "TRACKING_WITH_LOCAL_MAP":
            if self.map_initial_descriptors is None or self.map_points.shape[0] == 0 or len(kps) < self.min_features_for_pnp:
                self.prev_kps, self.prev_des = kps, des
                return self.current_R_w2c, self.current_t_w2c, viz_kps, viz_matches

            matches_to_map = self._match_features(self.map_initial_descriptors, des) # Match current des to map des
            viz_matches = matches_to_map # For visualization

            if len(matches_to_map) >= self.min_features_for_pnp:
                # queryIdx refers to map_initial_descriptors (and thus self.map_points)
                # trainIdx refers to current frame's kps/des
                object_pts_indices = [m.queryIdx for m in matches_to_map]
                image_pts_indices = [m.trainIdx for m in matches_to_map]

                object_points = self.map_points[object_pts_indices]
                image_points = np.float32([kps[i].pt for i in image_pts_indices]).reshape(-1,1,2)
                
                # Undistort image points for solvePnP
                image_points_undistorted = cv2.undistortPoints(image_points, self.K, self.dist_coeffs, P=self.K)
                if image_points_undistorted is None: # Check for undistortion failure
                    self.prev_kps, self.prev_des = kps, des
                    return self.current_R_w2c, self.current_t_w2c, viz_kps, viz_matches

                try:
                    success, rvec, tvec, pnp_inliers = cv2.solvePnPRansac(
                        object_points, image_points_undistorted, self.K, None, # Pass None for dist_coeffs as points are undistorted
                        iterationsCount=100, reprojectionError=8.0, confidence=0.99,
                        flags=cv2.SOLVEPNP_ITERATIVE # Using iterative method
                    )
                    if success and pnp_inliers is not None and len(pnp_inliers) >= self.min_features_for_pnp :
                        R_w2c_new, _ = cv2.Rodrigues(rvec)
                        t_w2c_new = tvec
                        
                        # The pose from solvePnP is camera_to_world. We need world_to_camera.
                        # R_c2w = R_w2c_new, t_c2w = t_w2c_new
                        # R_w2c = R_c2w.T, t_w2c = -R_c2w.T @ t_c2w
                        self.current_R_w2c = R_w2c_new.T 
                        self.current_t_w2c = -R_w2c_new.T @ t_w2c_new
                        
                        # Keyframe Selection
                        last_kf_R = self.keyframes[-1]['R']
                        last_kf_t = self.keyframes[-1]['t']
                        
                        # Note: current_R_w2c and current_t_w2c are world-in-camera.
                        # To compare with keyframe poses (which are also world-in-camera), direct comparison is fine.
                        trans_dist, rot_angle = self._calculate_camera_movement(
                            last_kf_R, last_kf_t, 
                            self.current_R_w2c, self.current_t_w2c
                        )

                        if trans_dist > self.min_kf_translation_dist or rot_angle > self.min_kf_rotation_angle:
                            print(f"New KeyFrame added. Dist: {trans_dist:.2f}m, Angle: {rot_angle:.2f}deg")
                            self.keyframes.append({
                                'R': self.current_R_w2c.copy(), 
                                't': self.current_t_w2c.copy(), 
                                'kps': kps, 
                                'des': des
                            })
                            # Placeholder: Future work would involve updating map_initial_descriptors and map_points
                            # based on this new keyframe and triangulating new points.
                            # For now, PnP will continue to use the initial map.
                    else:
                        # print(f"solvePnP failed or not enough inliers: {len(pnp_inliers) if pnp_inliers is not None else 'None'}")
                        pass # Keep previous pose if PnP fails
                except cv2.error as e:
                    # print(f"cv2.error in solvePnPRansac: {e}")
                    pass # Keep previous pose if PnP raises error
            
            self.prev_kps, self.prev_des = kps, des
            return self.current_R_w2c, self.current_t_w2c, viz_kps, viz_matches
        
        self.prev_kps, self.prev_des = kps, des
        return self.current_R_w2c, self.current_t_w2c, viz_kps, viz_matches


if __name__ == '__main__':
    print("SLAMFrontend Example Usage (Illustrative)")
    img_w, img_h = 640, 480
    dummy_K = np.array([[550, 0, img_w/2], [0, 550, img_h/2], [0, 0, 1]], dtype=np.float32)
    class DummyCameraParams:
        def get_K(self): return dummy_K
        def get_dist_coeffs(self): return np.zeros((5,1), dtype=np.float32) 
        def get_image_dimensions(self): return img_w, img_h

    cam_params = DummyCameraParams()
    slam_frontend = SLAMFrontend(camera_params=cam_params)
    
    # Create dummy frames
    frame1_bgr = np.random.randint(100, 255, (img_h, img_w, 3), dtype=np.uint8) # Brighter BG
    cv2.putText(frame1_bgr, "Frame 1 (KF1)", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,200), 2)

    # Simulate camera motion for KF2 (significant translation and some rotation for good baseline)
    R_f1_to_f2, _ = cv2.Rodrigues(np.array([0.05, -0.1, 0.02])) # Small rotation
    t_f1_to_f2 = np.array([[0.25], [0.05], [0.1]]) # Translation of 0.25m in X
    
    # Warp perspective for frame 2 to simulate rotation and translation
    H_f1_to_f2 = dummy_K @ np.hstack((R_f1_to_f2, t_f1_to_f2 / slam_frontend.initial_baseline_scale * 0.1 )) @ np.linalg.inv(dummy_K) # Approximation
    # Simplified: Use affine transform to simulate motion for dummy data robustness
    M_f2 = np.float32([[1, 0, 40], [0, 1, 10]]) # Shift by (40,10)
    frame2_bgr = cv2.warpAffine(frame1_bgr, M_f2, (img_w, img_h))
    # frame2_bgr = cv2.warpPerspective(frame1_bgr, H_f1_to_f2, (img_w,img_h)) # Can be unstable with random data
    cv2.putText(frame2_bgr, "Frame 2 (KF2)", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,200), 2)
    
    # Simulate further motion for frame 3 (tracking)
    M_f3 = np.float32([[1, 0, 15], [0, 1, 8]]) # Smaller shift relative to frame 2
    frame3_bgr = cv2.warpAffine(frame2_bgr, M_f3, (img_w, img_h))
    cv2.putText(frame3_bgr, "Frame 3 (Tracking)", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,200), 2)

    # Simulate motion for frame 4 (potential new KF)
    M_f4 = np.float32([[1, 0, 50], [0, 1, 20]]) # Larger shift relative to frame 3
    frame4_bgr = cv2.warpAffine(frame3_bgr, M_f4, (img_w, img_h))
    cv2.putText(frame4_bgr, "Frame 4 (New KF?)", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,200), 2)


    print("Processing first frame...")
    R_w2c_f1, t_w2c_f1, _, _ = slam_frontend.process_frame(frame1_bgr)
    print(f"State: {slam_frontend.state}, KFs: {len(slam_frontend.keyframes)}")

    print("\nProcessing second frame...")
    R_w2c_f2, t_w2c_f2, _, matches_f2 = slam_frontend.process_frame(frame2_bgr)
    print(f"State: {slam_frontend.state}, KFs: {len(slam_frontend.keyframes)}")
    print(f"Matches KF1-KF2: {len(matches_f2)}")
    if slam_frontend.map_points.shape[0] > 0:
        print(f"Map points created: {slam_frontend.map_points.shape[0]}")
        if slam_frontend.map_initial_descriptors is not None:
            print(f"Map initial descriptors stored: {slam_frontend.map_initial_descriptors.shape[0]}")
    else:
        print("Map points not created yet.")

    print("\nProcessing third frame (Tracking with Local Map)...")
    R_w2c_f3, t_w2c_f3, _, matches_f3 = slam_frontend.process_frame(frame3_bgr)
    print(f"State: {slam_frontend.state}, KFs: {len(slam_frontend.keyframes)}")
    print(f"Pose F3 (w2c): t = {t_w2c_f3.T}")
    print(f"Matches to Map: {len(matches_f3)}")


    print("\nProcessing fourth frame (Potential New KF)...")
    R_w2c_f4, t_w2c_f4, _, matches_f4 = slam_frontend.process_frame(frame4_bgr)
    print(f"State: {slam_frontend.state}, KFs: {len(slam_frontend.keyframes)}")
    print(f"Pose F4 (w2c): t = {t_w2c_f4.T}")
    print(f"Matches to Map: {len(matches_f4)}")


    print("\n--- Final State ---")
    print(f"Total Keyframes: {len(slam_frontend.keyframes)}")
    print(f"Total Map Points: {slam_frontend.map_points.shape[0]}")
    if slam_frontend.map_initial_descriptors is not None:
        print(f"Map Initial Descriptors: {slam_frontend.map_initial_descriptors.shape[0]}")

    # cv2.imshow("Frame 1", frame1_bgr)
    # cv2.imshow("Frame 2", frame2_bgr)
    # cv2.imshow("Frame 3", frame3_bgr)
    # cv2.imshow("Frame 4", frame4_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
