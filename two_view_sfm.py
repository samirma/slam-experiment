import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 2. Load camera calibration parameters ---
try:
    calibration_data = np.load("calibration_params.npz")
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']
    print("Calibration parameters loaded successfully.")
except FileNotFoundError:
    print("Error: calibration_params.npz not found. Please run calibrate_camera.py first.")
    exit()

# --- 3. Initialize ORB detector and Brute-Force Matcher ---
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- 4. Initialize variables ---
frame1_color = None
kps1 = None
des1 = None

# --- 5. Start video capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting camera for Two-View SfM...")

# --- 6. Main loop ---
while True:
    # --- a. Read a frame. Undistort it ---
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)

    # --- b. Convert to grayscale. Detect features and descriptors ---
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    kps_current, des_current = orb.detectAndCompute(gray, None)

    # --- c. Draw keypoints on the frame for visualization ---
    display_frame = cv2.drawKeypoints(undistorted_frame, kps_current, None, color=(0,255,0), flags=0)

    # --- d. Display the frame with a message ---
    message = "Press 'c' to capture first frame"
    if frame1_color is not None:
        message = "Press 'n' to capture second frame and process"
    cv2.putText(display_frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Two-View SfM", display_frame)

    # --- e. Key handling ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if frame1_color is None:
            frame1_color = undistorted_frame.copy() # Store the color frame
            kps1 = kps_current
            des1 = des_current
            print("First frame captured. Move camera and press 'n'.")
        else:
            print("First frame already captured. Press 'n' for second or 'q' to quit.")

    elif key == ord('n'):
        if frame1_color is not None and des_current is not None and des1 is not None:
            print("\nCapturing second frame and processing...")
            frame2_color = undistorted_frame.copy()
            kps2 = kps_current
            des2 = des_current

            # Match descriptors
            matches = bf.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:100] # Keep top 100
            print(f"Found {len(good_matches)} good matches.")

            if len(good_matches) < 10: # Need enough matches for E matrix
                print("Not enough good matches to proceed. Try again.")
                continue

            # Extract matched points
            pts1 = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Estimate Essential Matrix
            E, mask_e = cv2.findEssentialMat(pts1, pts2, mtx, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is None:
                print("Could not estimate Essential Matrix. Try different views.")
                continue
            print("Essential Matrix estimated.")

            # Recover Pose
            retval, R, t, mask_rp = cv2.recoverPose(E, pts1, pts2, mtx, mask=mask_e)
            
            if retval == 0 or R is None or t is None:
                print("Could not recover pose. Try different views.")
                continue
            
            num_inliers = np.sum(mask_rp > 0) if mask_rp is not None else 0
            print(f"Number of inliers from recoverPose: {num_inliers}")

            if num_inliers < 5: # Need a minimum number of inliers for triangulation
                 print("Not enough inliers after recoverPose. Try different views.")
                 continue
            
            print("Pose recovered.")
            print("R:\n", R)
            print("t:\n", t)

            # Form projection matrices
            P1 = mtx @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = mtx @ np.hstack((R, t))

            # Triangulate points
            # Ensure pts1 and pts2 are (2,N)
            pts1_tri = pts1.reshape(-1, 2).T
            pts2_tri = pts2.reshape(-1, 2).T
            
            points4D_hom = cv2.triangulatePoints(P1, P2, pts1_tri, pts2_tri)

            # Convert to Cartesian
            points3D = points4D_hom[:3, :] / points4D_hom[3, :]
            
            # Filter points based on mask_rp from recoverPose
            # mask_rp should be 1D. If it's (N,1), ravel it.
            if mask_rp is not None:
                valid_points_mask = mask_rp.ravel() > 0
                points3D_filtered = points3D[:, valid_points_mask].T # Transpose to get (N,3)
            else: # Should not happen if recoverPose was successful with a mask
                points3D_filtered = points3D.T 

            print(f"Successfully triangulated {points3D_filtered.shape[0]} 3D points.")
            if points3D_filtered.shape[0] > 0:
                print("First 5 triangulated 3D points (X, Y, Z):\n", points3D_filtered[:5, :])

                # --- Visualization of 3D points ---
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                
                # Scatter plot the 3D points
                # points3D_filtered is of shape (N, 3)
                ax.scatter(points3D_filtered[:,0], points3D_filtered[:,1], points3D_filtered[:,2], c='r', marker='o', s=5)
                
                # Plot camera positions
                # First camera (origin)
                ax.scatter(0, 0, 0, c='g', marker='^', s=100, label='Cam1 (Origin)')
                # Second camera position (derived from R and t)
                # The camera center is -R.T @ t
                cam2_center = -R.T @ t
                ax.scatter(cam2_center[0], cam2_center[1], cam2_center[2], c='b', marker='^', s=100, label='Cam2')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'{points3D_filtered.shape[0]} 3D Points Reconstructed')
                ax.legend()
                
                # Optional: Set axis limits
                # ax.set_xlim([-5, 5])
                # ax.set_ylim([-5, 5])
                # ax.set_zlim([0, 10])
                
                plt.show()
            else:
                print("No 3D points to visualize.")

            # Optional: Visualize matches (shown before 3D plot)
            img_matches = cv2.drawMatches(frame1_color, kps1, frame2_color, kps2, 
                                          [m for i, m in enumerate(good_matches) if mask_rp is None or (mask_rp[i] > 0 if mask_rp.ndim > 1 else mask_rp[i] > 0) ], # only draw inlier matches
                                          None, 
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Good Matches and Inliers", img_matches)
            cv2.waitKey(0) # Wait until a key is pressed, then the 3D plot will show if points exist.

            break # Break the loop after processing

        elif frame1_color is None:
            print("Please capture the first frame ('c') before the second ('n').")
        else:
            print("Could not detect features in the current frame for second view.")


    elif key == ord('q'):
        print("Quitting...")
        break

# --- 7. Release capture, destroy windows ---
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")
