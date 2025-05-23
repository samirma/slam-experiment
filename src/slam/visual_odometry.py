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
        matches = sorted(matches, key=lambda x: x.distance)
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
        kps, des = self._extract_features(frame_gray)
        viz_kps = kps if kps is not None else []
        viz_matches = []

        if self.prev_frame_gray is None or self.prev_kps is None or self.prev_des is None:
            self.prev_frame_gray = frame_gray
            self.prev_kps = kps
            self.prev_des = des
            return self.cur_R, self.cur_t, viz_kps, viz_matches

        matches = self._match_features(self.prev_des, des)
        viz_matches = matches

        if len(matches) >= self.min_features_for_tracking:
            pts_prev_indices = [m.queryIdx for m in matches]
            pts_curr_indices = [m.trainIdx for m in matches]

            pts_prev = np.float32([self.prev_kps[i].pt for i in pts_prev_indices]).reshape(-1, 1, 2)
            pts_curr = np.float32([kps[i].pt for i in pts_curr_indices]).reshape(-1, 1, 2)

            # Undistort points using K and dist_coeffs. P=self.K projects them back to the image plane
            # as if seen by a perfect pinhole camera with matrix K.
            pts_prev_undistorted = cv2.undistortPoints(pts_prev, self.K, self.dist_coeffs, P=self.K)
            pts_curr_undistorted = cv2.undistortPoints(pts_curr, self.K, self.dist_coeffs, P=self.K)
            
            # Ensure points are continuous and have the correct depth (float32 or float64)
            # cv2.undistortPoints with P=K should return float32 points if K is float32.
            # Reshape to (N, 2) for findEssentialMat if needed, or ensure it handles (N,1,2)
            # findEssentialMat expects points2 (current), points1 (previous)
            E, e_mask = cv2.findEssentialMat(pts_curr_undistorted, pts_prev_undistorted,
                                             cameraMatrix=self.K,
                                             method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is not None and e_mask is not None:
                pts_curr_undistorted_inliers = pts_curr_undistorted[e_mask.ravel() == 1]
                pts_prev_undistorted_inliers = pts_prev_undistorted[e_mask.ravel() == 1]

                if len(pts_curr_undistorted_inliers) >= self.min_features_for_tracking:
                    # Recover relative pose. Since points are already undistorted,
                    # pass None for distortion_coefficients.
                    num_inliers, R_rel, t_rel, rp_mask_recover_pose = cv2.recoverPose(
                        E,
                        pts_curr_undistorted_inliers, # Current frame points
                        pts_prev_undistorted_inliers, # Previous frame points
                        self.K,
                        None  # IMPORTANT: Distortion coefficients set to None as points are already undistorted
                        # mask=out_mask_recover_pose # Optional output mask for recoverPose inliers
                    )
                    
                    # Check if recoverPose was successful and returned enough inliers
                    # num_inliers here is the count of points consistent with the recovered pose.
                    if R_rel is not None and t_rel is not None and num_inliers >= self.min_features_for_tracking:
                        # Update cumulative pose: R_w2curr = R_prev2curr @ R_w2prev
                        self.cur_R = R_rel @ self.cur_R
                        # t_w_origin_in_curr = R_prev2curr @ t_w_origin_in_prev + t_curr_origin_in_prev_frame
                        self.cur_t = R_rel @ self.cur_t + t_rel
                    else:
                        # print(f"VO: recoverPose failed or not enough inliers ({num_inliers}).")
                        pass # Keep previous pose
                else:
                    # print(f"VO: Not enough inliers after findEssentialMat ({len(pts_curr_undistorted_inliers)}).")
                    pass # Keep previous pose
            else:
                # print("VO: Essential Matrix estimation failed.")
                pass # Keep previous pose
        else:
            # print(f"VO: Not enough initial matches ({len(matches)}).")
            pass # Keep previous pose

        self.prev_frame_gray = frame_gray
        self.prev_kps = kps
        self.prev_des = des

        return self.cur_R, self.cur_t, viz_kps, viz_matches


if __name__ == '__main__':
    print("VisualOdometry Example Usage (Illustrative)")
    img_w, img_h = 640, 480
    dummy_K = np.array([[550, 0, img_w/2], [0, 550, img_h/2], [0, 0, 1]], dtype=np.float32)
    class DummyCameraParams:
        def get_K(self): return dummy_K
        def get_dist_coeffs(self): return np.zeros((5,1), dtype=np.float32) # Use (5,1) or (1,5)
        def get_image_dimensions(self): return img_w, img_h

    cam_params = DummyCameraParams()
    vo = VisualOdometry(camera_params=cam_params)
    frame1_bgr = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    frame2_bgr = cv2.warpAffine(frame1_bgr, M, (img_w, img_h))

    print("Processing first frame...")
    R1, t1, kps1, matches1 = vo.process_frame(frame1_bgr)
    print(f"Pose after frame 1: R = \n{R1}\n t = \n{t1.T}")

    print("\nProcessing second frame...")
    R2, t2, kps2, matches2 = vo.process_frame(frame2_bgr)
    print(f"Pose after frame 2: R = \n{R2}\n t = \n{t2.T}")
    print(f"Matches found: {len(matches2)}")
    print("\nNote: Dummy frames are not ideal for robust VO testing.")
