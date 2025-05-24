import cv2
import numpy as np

# --- 1. Import cv2 and numpy ---
# Done at the top

# --- 2. Load camera calibration parameters ---
try:
    calibration_data = np.load("calibration_params.npz")
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']
    print("Calibration parameters loaded successfully.")
except FileNotFoundError:
    print("Error: calibration_params.npz not found. Please run calibrate_camera.py first.")
    exit()

# --- 3. Initialize ORB feature detector ---
orb = cv2.ORB_create(nfeatures=1000)

# --- 4. Initialize Brute-Force Matcher ---
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- 5. Initialize variables for previous frame's keypoints and descriptors ---
prev_kps = None
prev_des = None

# --- 6. Start video capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting camera for feature tracking...")
print("Press 'q' to quit.")

# --- 7. Main loop ---
while True:
    # --- a. Read a frame ---
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # --- b. Undistort the frame ---
    # Using mtx as the new camera matrix for simplicity. 
    # For better results, one might use cv2.getOptimalNewCameraMatrix.
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)

    # --- c. Convert to grayscale ---
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

    # --- d. Detect keypoints and compute descriptors ---
    current_kps, current_des = orb.detectAndCompute(gray, None)

    # --- e. Create an output image ---
    # Drawing on the color undistorted frame
    output_img = undistorted_frame.copy() 

    # --- f. If not the first frame, perform matching ---
    if prev_kps is not None and prev_des is not None and current_des is not None:
        # i. Match descriptors
        matches = bf.match(prev_des, current_des)

        # ii. Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # iii. Select good matches (e.g., top 50)
        num_good_matches = 50
        good_matches = matches[:num_good_matches]

        # iv. Draw good matches
        # We will draw circles on current keypoints that have a good match
        if current_kps and good_matches:
            for match in good_matches:
                # match.trainIdx refers to the index in current_kps/current_des
                # match.queryIdx refers to the index in prev_kps/prev_des
                kp_index_current = match.trainIdx 
                if kp_index_current < len(current_kps):
                    pt = tuple(map(int, current_kps[kp_index_current].pt))
                    cv2.circle(output_img, pt, 5, (0, 255, 0), 1) # Green circle

        # Alternative: Draw lines using cv2.drawMatches (more complex to show only current frame)
        # For simplicity, the circle drawing method is used as requested.
        # If you wanted to draw lines between previous and current frame matches,
        # you would need the previous frame image as well.
        # For example:
        # img_matches = cv2.drawMatches(prev_gray_bgr, prev_kps, gray_bgr, current_kps, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # output_img = img_matches # replace output_img if using this

    # --- g. Display output_img ---
    cv2.imshow("Feature Tracking", output_img)

    # --- h. Update previous keypoints and descriptors ---
    prev_kps = current_kps
    prev_des = current_des
    # Store a BGR version of the gray image if you plan to use cv2.drawMatches
    # prev_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) 

    # --- i. Handle key presses ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# --- 8. Release camera and destroy windows ---
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")
