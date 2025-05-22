"""
Runs a Monocular Visual Odometry (VO) system.

This script captures frames from a camera, processes them using the `VisualOdometry`
class, and displays the estimated camera trajectory along with feature matches.

Key functionalities:
- Initializes the camera and loads camera parameters (from calibration or defaults).
- Initializes the `VisualOdometry` module.
- In a loop:
    - Captures a frame.
    - Processes the frame with `VisualOdometry` to get the current pose.
    - Visualizes detected keypoints and matches between current and previous frames.
    - Draws the estimated 2D trajectory (X-Z plane) of the camera.
- Allows resetting the trajectory visualization.
"""
import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.camera.camera import MonocularCamera
from src.slam import VisualOdometry # Uses __init__.py
from src.utils import CameraParams # Uses __init__.py

def main():
    """
    Main function to run Monocular Visual Odometry, display features, and trajectory.
    """
    print("Starting Visual Odometry application...")
    camera = None
    
    # --- Configuration ---
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    # FX = 550.0 # Example value
    # FY = 550.0 # Example value
    CALIBRATION_FILE = "data/camera_calibration.yaml" # Path to the calibration file
    
    TRAJECTORY_IMG_WIDTH = 800
    TRAJECTORY_IMG_HEIGHT = 600
    TRAJECTORY_SCALE = 150 # Pixels per meter (adjust based on expected motion scale)
    TRAJECTORY_OFFSET_X = TRAJECTORY_IMG_WIDTH // 2
    TRAJECTORY_OFFSET_Y = TRAJECTORY_IMG_HEIGHT - 100 # Start near bottom-center

    try:
        print("Initializing camera...")
        camera = MonocularCamera(0) # Or specify a video file path
        # Ensure camera is providing frames of the expected size, or adjust CameraParams
        # For simplicity, we assume camera provides IMAGE_WIDTH x IMAGE_HEIGHT
        print(f"Camera initialized. Expected frame size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")

        print("Initializing Camera Parameters...")
        # Use actual frame width/height if they can be read from camera before VO init
        # success_init, frame_init = camera.get_frame()
        # if success_init:
        #    IMAGE_HEIGHT, IMAGE_WIDTH, _ = frame_init.shape
        # else:
        #    print("Could not get initial frame to set image dimensions, using defaults.")
        # cam_params = CameraParams(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, fx=FX, fy=FY)
        cam_params = CameraParams(calibration_file_path=CALIBRATION_FILE, 
                                  default_image_width=IMAGE_WIDTH, 
                                  default_image_height=IMAGE_HEIGHT)
        # Update IMAGE_WIDTH and IMAGE_HEIGHT from actual loaded params if they changed
        IMAGE_WIDTH, IMAGE_HEIGHT = cam_params.get_image_dimensions()
        print(f"Camera parameters initialized using '{CALIBRATION_FILE}'. Loaded K=\n{cam_params.get_K()}")
        print(f"Image dimensions from CameraParams: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")


        print("Initializing Visual Odometry...")
        vo = VisualOdometry(camera_params=cam_params)
        print("Visual Odometry initialized.")

        # Trajectory visualization
        trajectory_img = np.zeros((TRAJECTORY_IMG_HEIGHT, TRAJECTORY_IMG_WIDTH, 3), dtype=np.uint8)
        cv2.putText(trajectory_img, "Trajectory (X-Z plane, Y is up in world)", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Store previous frame for drawing matches
        prev_frame_display = None
        prev_kps_display = None

        print("Press 'q' to quit.")
        cv2.namedWindow('VO Output', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Trajectory', cv2.WINDOW_AUTOSIZE)

        while True:
            success, frame_bgr = camera.get_frame()

            if not success:
                print("Failed to capture frame or end of video stream.")
                if camera.camera_id != 0:
                    print("End of video file.")
                break
            
            # Resize if camera output doesn't match configured size (optional, better to configure CameraParams correctly)
            if frame_bgr.shape[1] != IMAGE_WIDTH or frame_bgr.shape[0] != IMAGE_HEIGHT:
                 frame_bgr = cv2.resize(frame_bgr, (IMAGE_WIDTH, IMAGE_HEIGHT))


            # --- Visual Odometry Processing ---
            current_R, current_t, kps, matches = vo.process_frame(frame_bgr)
            
            # --- Visualization ---
            # Draw keypoints on the current frame
            frame_with_kps = cv2.drawKeypoints(frame_bgr, kps, None, color=(0, 255, 0), 
                                               flags=cv2.DrawMatchesFlags_DEFAULT)

            # Draw matches if previous frame and keypoints are available
            # Note: vo.prev_kps refers to keypoints in vo.prev_frame_gray (which is k-1)
            # and kps are for current frame_gray (k). Matches are between these.
            if prev_frame_display is not None and vo.prev_kps is not None and len(matches) > 0:
                # vo.prev_kps are the 'query' keypoints, kps are the 'train' keypoints for the matches
                img_matches = cv2.drawMatches(prev_frame_display, vo.prev_kps, # Use vo.prev_kps as they correspond to vo.prev_des
                                              frame_bgr, kps, 
                                              matches[:50], None, # Draw top 50 matches
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                output_display = img_matches
            else:
                output_display = frame_with_kps

            cv2.imshow('VO Output', output_display)

            # Update previous frame and keypoints for next iteration's match drawing
            prev_frame_display = frame_bgr.copy() # Store the color frame for display
            # prev_kps_display = kps # kps are for the current frame, which becomes prev_kps in next iteration

            # Trajectory Visualization
            # current_t is the position of the camera in the world (relative to start)
            # X-axis is typically to the right, Y-axis down, Z-axis forward in image coordinates.
            # In a common world coordinate system for VO/SLAM: X right, Y up, Z forward.
            # OpenCV's recoverPose gives t with X right, Y down, Z forward (camera system).
            # If self.cur_R, self.cur_t is camera pose in world (R: world_to_cam, t: cam_pos_in_world)
            # The translation t from recoverPose is the camera's *motion* in the *previous camera's coordinate system*.
            # self.cur_t = self.cur_t + self.cur_R @ t_motion where self.cur_R is R_world_to_prev_cam
            # Let's assume self.cur_t in VisualOdometry is the camera's *position* in the world frame.
            # And the world frame is X right, Y down, Z forward (aligned with first camera).
            # If we want X right, Y up, Z forward for world:
            # Initial camera looks along Z+. X right, Y down.
            # If VO's self.cur_t is [x, y, z] where x is right, y is down, z is forward:
            world_x = current_t[0, 0]
            world_y_opencv = current_t[1, 0] # Y from OpenCV is typically 'down'
            world_z = current_t[2, 0]

            # To draw on image (origin top-left):
            # X on image: maps to world X
            # Y on image: maps to world Z (depth) or world -Y (if Y is up)
            # Let's plot X-Z plane (top-down view, Y is effectively 'up' out of the plane)
            draw_x = int(TRAJECTORY_OFFSET_X + world_x * TRAJECTORY_SCALE)
            draw_y = int(TRAJECTORY_OFFSET_Y - world_z * TRAJECTORY_SCALE) # Use -Z if Z is forward and Y in image is down

            # Draw current position
            cv2.circle(trajectory_img, (draw_x, draw_y), 3, (0, 255, 0), -1) # Green for current position

            # Draw lines connecting previous positions (if you store history)
            # For simplicity, just draw the current point.
            # To draw lines: store previous (draw_x, draw_y) and use cv2.line()

            cv2.imshow('Trajectory', trajectory_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'): # Reset trajectory
                print("Resetting trajectory plot and VO state (if implemented in VO)...")
                trajectory_img.fill(0) # Clear image
                cv2.putText(trajectory_img, "Trajectory (X-Z plane, Y is up in world)", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                # Optionally, re-initialize VO or add a reset method to it
                # vo = VisualOdometry(camera_params=cam_params) # This would reset VO state

    except IOError as e:
        print(f"Camera/IO Error: {e}")
    except ImportError as e:
        print(f"ImportError: {e}. Ensure 'src' is in PYTHONPATH or script is run from project root.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if camera is not None and camera.cap.isOpened():
            camera.release()
            print("Camera released.")
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")
        print("Application finished.")

if __name__ == "__main__":
    main()
