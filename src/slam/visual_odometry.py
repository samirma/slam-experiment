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


class VisualOdometry:
    """
    Implements a basic Monocular Visual Odometry (VO) system using ORB features.

    This class tracks the camera's pose (rotation and translation) relative to its starting
    position by analyzing sequential image frames. It detects and matches features (ORB)
    between frames to estimate the motion.

    The coordinate system convention used for camera pose (self.cur_R, self.cur_t) is:
    - `self.cur_R`: Rotation matrix representing the orientation of the world with respect
                     to the current camera frame (R_world_to_camera).
    - `self.cur_t`: Translation vector representing the position of the world origin
                     in the current camera's coordinate system (t_world_to_camera).
    Alternatively, to get camera_in_world pose:
    - R_camera_in_world = self.cur_R.T
    - t_camera_in_world = -self.cur_R.T @ self.cur_t
    The VO process updates `self.cur_R` and `self.cur_t` to reflect the transformation
    from the world frame (defined by the first camera pose) to the current camera frame.
    """
    def __init__(self, camera_params: CameraParams):
        """
        Initializes the Visual Odometry system.

        Args:
            camera_params (CameraParams): An object containing camera intrinsic parameters
                                          (matrix K and distortion coefficients).
        """
        self.camera_params: CameraParams = camera_params
        self.K: np.ndarray = self.camera_params.get_K()
        self.dist_coeffs: np.ndarray = self.camera_params.get_dist_coeffs()

        # Feature detection and matching
        self.orb: cv2.ORB = cv2.ORB_create(nfeatures=1500, # Increased features slightly
                                           scaleFactor=1.2,
                                           nlevels=8,
                                           edgeThreshold=31,
                                           firstLevel=0,
                                           WTA_K=2,
                                           scoreType=cv2.ORB_HARRIS_SCORE, # Harris score can be more stable
                                           patchSize=31,
                                           fastThreshold=20)
        self.matcher: cv2.BFMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # State variables
        # cur_R represents R_world_to_camera (orientation of world in current camera frame)
        self.cur_R: np.ndarray = np.eye(3, dtype=np.float32) 
        # cur_t represents t_world_to_camera (position of world origin in current camera frame)
        self.cur_t: np.ndarray = np.zeros((3, 1), dtype=np.float32)

        self.prev_frame_gray: np.ndarray | None = None
        self.prev_kps: list[cv2.KeyPoint] | None = None
        self.prev_des: np.ndarray | None = None

        self.min_features_for_tracking: int = 10 # Minimum number of good matches for pose recovery

    def _extract_features(self, frame_gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """
        Detects ORB keypoints and computes their descriptors from a grayscale frame.

        Args:
            frame_gray (numpy.ndarray): Grayscale input image (H, W).

        Returns:
            tuple[list[cv2.KeyPoint], numpy.ndarray | None]: 
                - keypoints (list[cv2.KeyPoint]): A list of detected keypoints.
                - descriptors (numpy.ndarray | None): An array of ORB descriptors (Nx32 bytes),
                                                      or None if no keypoints are found.
        """
        kps, des = self.orb.detectAndCompute(frame_gray, None)
        return kps, des

    def _match_features(self, des_prev: np.ndarray, des_curr: np.ndarray) -> list[cv2.DMatch]:
        """
        Matches ORB descriptors from the previous and current frames using a Brute-Force matcher
        with Hamming distance and cross-check.

        Args:
            des_prev (numpy.ndarray): Descriptors from the previous frame.
            des_curr (numpy.ndarray): Descriptors from the current frame.

        Returns:
            list[cv2.DMatch]: A list of good matches (cv2.DMatch objects), sorted by distance.
                              Returns an empty list if input descriptors are invalid or no matches are found.
        """
        if des_prev is None or des_curr is None or des_prev.shape[0] == 0 or des_curr.shape[0] == 0:
            return []
        
        matches = self.matcher.match(des_prev, des_curr)
        
        # Sort them in the order of their distance (lower distance is better).
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Optional: Further filtering if needed, e.g., ratio test if not using crossCheck.
        # For BFMatcher with crossCheck=True, matches are already quite good.
        # We can limit the number of matches to consider for performance if necessary.
        # For example, take top N_MATCHES:
        # N_MATCHES = 100
        # matches = matches[:N_MATCHES]
        
        return matches

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[cv2.KeyPoint], list[cv2.DMatch]]:
        """
        Processes a new BGR frame to estimate the camera's motion relative to the previous frame
        and updates the cumulative camera pose (world-to-camera).

        Args:
            frame_bgr (numpy.ndarray): Input color image (BGR format, H, W, C).

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, list[cv2.KeyPoint], list[cv2.DMatch]]:
                - R_w2c (numpy.ndarray): The updated 3x3 rotation matrix representing the
                                         orientation of the world in the current camera frame.
                - t_w2c (numpy.ndarray): The updated 3x1 translation vector representing the
                                         position of the world origin in the current camera frame.
                - current_keypoints (list[cv2.KeyPoint]): Keypoints detected in the current frame (for visualization).
                - matches (list[cv2.DMatch]): Matches found between current and previous frame (for visualization).
        """
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # TODO: Consider applying histogram equalization to frame_gray if lighting conditions vary significantly.
        # frame_gray = cv2.equalizeHist(frame_gray)
        
        kps, des = self._extract_features(frame_gray)

        # For visualization purposes, always return current keypoints (even if tracking fails)
        viz_kps = kps if kps is not None else []
        viz_matches = []

        if self.prev_frame_gray is None or self.prev_kps is None or self.prev_des is None:
            # This is the first frame, or previous frame had no features.
            # Store its features and return the current (initial or last known) pose.
            self.prev_frame_gray = frame_gray
            self.prev_kps = kps
            self.prev_des = des
            return self.cur_R, self.cur_t, viz_kps, viz_matches

        # Match features with the previous frame
        matches = self._match_features(self.prev_des, des)
        viz_matches = matches # Store for visualization output

        if len(matches) >= self.min_features_for_tracking:
            # Get corresponding 2D points for Essential Matrix estimation
            # queryIdx refers to the "query" descriptors (self.prev_des in our case)
            # trainIdx refers to the "train" descriptors (current 'des')
            pts1_indices = [m.queryIdx for m in matches]
            pts2_indices = [m.trainIdx for m in matches]

            pts1 = np.float32([self.prev_kps[i].pt for i in pts1_indices]).reshape(-1, 1, 2)
            pts2 = np.float32([kps[i].pt for i in pts2_indices]).reshape(-1, 1, 2)

            # Estimate Essential Matrix
            # Note: cv2.findEssentialMat expects points from the current frame (pts2) first,
            # Get corresponding 2D points from previous and current keypoints for Essential Matrix estimation
            # queryIdx refers to descriptors from self.prev_des (previous frame)
            # trainIdx refers to descriptors from des (current frame)
            pts_prev_indices = [m.queryIdx for m in matches]
            pts_curr_indices = [m.trainIdx for m in matches]

            pts_prev = np.float32([self.prev_kps[i].pt for i in pts_prev_indices]).reshape(-1, 1, 2)
            pts_curr = np.float32([kps[i].pt for i in pts_curr_indices]).reshape(-1, 1, 2)
            
            # Estimate Essential Matrix (E)
            # E relates corresponding points in two images assuming pinhole camera model and K is known.
            # Points are expected as: current_points, previous_points
            E, e_mask = cv2.findEssentialMat(pts_curr, pts_prev, 
                                             cameraMatrix=self.K, distCoeffs=self.dist_coeffs,
                                             method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is not None and e_mask is not None:
                # Recover relative pose (R_rel, t_rel) from E.
                # R_rel: Rotation from previous camera frame to current camera frame.
                # t_rel: Translation of current camera origin relative to previous camera origin,
                #        expressed in the *previous* camera's coordinate system.
                # Pass the inlier points (pts_curr[e_mask.ravel()==1] and pts_prev[e_mask.ravel()==1]) to recoverPose.
                num_inliers, R_rel, t_rel, rp_mask = cv2.recoverPose(E, 
                                                                     points1=pts_curr[e_mask.ravel() == 1], 
                                                                     points2=pts_prev[e_mask.ravel() == 1], 
                                                                     cameraMatrix=self.K, 
                                                                     distCoeffs=self.dist_coeffs
                                                                    )

                if num_inliers >= self.min_features_for_tracking: # Check if enough inliers for a stable pose
                    # Update the cumulative camera pose (world-to-camera)
                    # self.cur_R is R_world_to_prev_cam
                    # self.cur_t is t_world_origin_in_prev_cam
                    # R_rel is R_prev_cam_to_curr_cam
                    # t_rel is t_curr_cam_origin_in_prev_cam (motion of current cam wrt prev, in prev cam's coord system)
                    
                    # New world-to-camera rotation: R_w2curr = R_prev2curr @ R_w2prev
                    self.cur_R = R_rel @ self.cur_R
                    # New world-to-camera translation: t_w_origin_in_curr = R_prev2curr @ t_w_origin_in_prev - R_prev2curr @ t_prev_origin_in_curr
                    # This is equivalent to: t_w_origin_in_curr = R_prev2curr @ t_w_origin_in_prev + t_curr_origin_in_prev_frame_neg
                    # Let's use the standard update:
                    # If self.cur_R and self.cur_t represent the transformation from the world to the camera frame (P_camera = self.cur_R @ P_world + self.cur_t)
                    # then the update rule for (R_rel, t_rel) being the transformation from previous to current camera view is:
                    # self.cur_t = R_rel @ self.cur_t + t_rel  -- This seems off.
                    #
                    # Let's re-verify the pose update logic based on class docstring convention:
                    # self.cur_R is R_world_to_camera (R_w2c)
                    # self.cur_t is t_world_origin_in_camera (t_w_in_c)
                    # R_rel is R_prev_to_curr
                    # t_rel is t_curr_origin_in_prev_frame (motion of current camera wrt previous camera, in prev_cam coords)
                    
                    # Update R_w2c: R_w2curr = R_prev2curr @ R_w2prev
                    # This means: self.cur_R = R_rel @ self.cur_R (This seems correct)

                    # Update t_w_in_c:
                    # P_curr = R_rel @ P_prev + t_rel (where t_rel is t_curr_cam_origin_in_prev_cam_coords)
                    # P_curr = R_rel @ (R_w2prev @ P_world + t_w_in_prev) + t_rel
                    # P_curr = (R_rel @ R_w2prev) @ P_world + (R_rel @ t_w_in_prev + t_rel)
                    # So, new_t_w_in_c = R_rel @ self.cur_t + t_rel
                    self.cur_t = R_rel @ self.cur_t + t_rel
                    # Note: The scale of t_rel is ambiguous in monocular VO without a known scale (e.g. from depth sensor or map).
                    # Here we assume t_rel has a consistent (but arbitrary) scale.
                else:
                    # Not enough inliers from recoverPose, keep previous pose.
                    # print(f"VO: Tracking lost - not enough inliers from recoverPose ({num_inliers}).")
                    pass # viz_matches will be empty or from findEssentialMat if that's preferred
            else:
                # Essential Matrix estimation failed, keep previous pose.
                # print("VO: Tracking lost - Essential Matrix estimation failed.")
                pass
        else: # Not enough matches from feature matching
            # print(f"VO: Tracking lost - not enough initial matches ({len(matches)}).")
            pass

        # Update previous frame and features for the next iteration
        self.prev_frame_gray = frame_gray
        self.prev_kps = kps
        self.prev_des = des

        return self.cur_R, self.cur_t, viz_kps, viz_matches


if __name__ == '__main__':
    # This block provides an example of how to use the VisualOdometry class.
    # It's illustrative and requires dummy data or a live camera feed to be fully functional.
    # Example Usage (Illustrative)
    print("VisualOdometry Example Usage (Illustrative)")

    # 1. Setup Camera Parameters (dummy)
    img_w, img_h = 640, 480
    # A more realistic K for 640x480 might have fx, fy around 500-700
    # For example, if FoV is ~60 degrees, fx = width / (2 * tan(fov_rad / 2))
    # fx = 640 / (2 * np.tan(np.deg2rad(60)/2)) ~= 554
    dummy_K = np.array([[550, 0, img_w/2],
                        [0, 550, img_h/2],
                        [0, 0, 1]], dtype=np.float32)
    
    class DummyCameraParams: # Minimal mock for CameraParams
        def get_K(self): return dummy_K
        def get_dist_coeffs(self): return np.zeros((4,1), dtype=np.float32)
        def get_image_dimensions(self): return img_w, img_h

    cam_params = DummyCameraParams()
    vo = VisualOdometry(camera_params=cam_params)

    # 2. Create dummy frames (e.g., two identical frames for simplicity of running)
    # In a real scenario, these would be sequential frames from a camera.
    frame1_bgr = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
    # Simulate a slight translation for the second frame for testing
    # (Actual feature matching would require more distinct features)
    M = np.float32([[1, 0, 5], [0, 1, 5]]) # 5 pixel shift
    frame2_bgr = cv2.warpAffine(frame1_bgr, M, (img_w, img_h))


    print("Processing first frame...")
    R1, t1, kps1, matches1 = vo.process_frame(frame1_bgr)
    print(f"Pose after frame 1: R = \n{R1}\n t = \n{t1.T}")
    print(f"Keypoints in frame 1: {len(kps1)}")

    # For the sake of this example, let's ensure prev_des is not None for matching
    if vo.prev_des is None: # Should be set by process_frame
        print("Warning: prev_des not set after first frame, which is unexpected.")


    print("\nProcessing second frame...")
    R2, t2, kps2, matches2 = vo.process_frame(frame2_bgr) # This frame would be matched against frame1
    print(f"Pose after frame 2: R = \n{R2}\n t = \n{t2.T}")
    print(f"Keypoints in frame 2: {len(kps2)}")
    print(f"Matches found between frame 1 and 2: {len(matches2)}")

    if len(matches2) > 0:
        # To visualize matches, you need frame1_gray, kps1 and frame2_gray, kps2
        # frame1_gray_example = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY)
        # frame2_gray_example = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY)
        # img_matches_example = cv2.drawMatches(frame1_gray_example, kps1,
        #                                       frame2_gray_example, kps2,
        #                                       matches2[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow("Matches example", img_matches_example)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("Matches visualization would require storing previous gray frame and kps in this example script.")
    
    print("\nNote: The dummy frames used here are not ideal for robust VO testing.")
    print("Real-world sequences with distinct features and motion are needed for meaningful results.")
