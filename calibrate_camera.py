import cv2
import numpy as np

# Define chessboard parameters
CHECKERBOARD = (6, 9)  # (inner corners per row, col)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Initialize lists to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting camera. Look for a chessboard pattern.")
print("Press 'c' to capture calibration image when corners are detected.")
print("Press 'q' to quit and perform calibration (min 10 images needed).")

captured_images_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Try to find chessboard corners
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    display_frame = frame.copy() # Work on a copy to display messages

    if ret_corners:
        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw the corners on the frame
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners2, ret_corners)
        cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "No corners found. Adjust camera or pattern.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, "Press 'q' to quit.", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Camera Calibration", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if ret_corners:
            objpoints.append(objp)
            imgpoints.append(corners2)
            captured_images_count += 1
            print(f"Captured image {captured_images_count}")
            # Briefly show the captured frame with corners
            cv2.imshow("Captured", display_frame)
            cv2.waitKey(500) # Display for 0.5 seconds
            cv2.destroyWindow("Captured")

        else:
            print("No corners detected in the current frame. Cannot capture.")
    elif key == ord('q'):
        print("Quitting capture mode.")
        break

# Perform calibration if enough points were captured
if captured_images_count >= 10:
    print(f"\nPerforming calibration with {captured_images_count} images...")
    # gray.shape[::-1] gives (width, height) which is what calibrateCamera expects for imageSize
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("\nCamera Calibration Successful!")
        print("\nCamera Matrix (mtx):\n", mtx)
        print("\nDistortion Coefficients (dist):\n", dist)

        # Save calibration data
        np.savez("calibration_params.npz", mtx=mtx, dist=dist)
        print("\nCalibration parameters saved to 'calibration_params.npz'")
        
        # Example of undistorting an image (optional, for verification)
        # undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)
        # cv2.imshow("Original vs Undistorted", np.hstack((frame, undistorted_frame)))
        # cv2.waitKey(0)

    else:
        print("\nError: Camera calibration failed.")
else:
    print(f"\nCalibration requires at least 10 images, but only {captured_images_count} were captured.")
    print("Aborting calibration.")

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")
