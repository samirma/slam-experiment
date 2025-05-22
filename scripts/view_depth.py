"""
Script to display real-time depth estimation from a monocular camera feed.

This script captures video frames using `MonocularCamera`, estimates depth for each
frame using `MiDaSDepthEstimator`, and then visualizes both the original camera
feed and the colorized depth map side-by-side.

Key functionalities:
- Initializes the camera and the MiDaS depth estimation model.
- In a loop:
    - Captures a frame.
    - Converts the frame to RGB (from BGR).
    - Estimates depth using the MiDaS model.
    - Normalizes and colorizes the depth map for visualization.
    - Displays the original frame and the depth map.
- Allows saving the current frame and depth map by pressing 's'.
- Calculates and displays FPS (Frames Per Second).
"""
import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.camera.camera import MonocularCamera
from src.depth_estimation import MiDaSDepthEstimator # Uses __init__.py

def main():
    """
    Main function to capture video, estimate depth, and display results.
    """
    print("Starting application...")
    camera = None
    try:
        print("Initializing camera...")
        camera = MonocularCamera(0) # Or specify a video file path
        print("Camera initialized successfully.")

        print("Initializing MiDaS Depth Estimator...")
        # You can choose the model URL here.
        # Smaller model: "https://tfhub.dev/intel/midas/v2_1_small/1"
        # Larger model (default in class): "https://tfhub.dev/intel/midas/v2_1/1"
        midas = MiDaSDepthEstimator() # Or MiDaSDepthEstimator(model_url="...")
        print("MiDaS Depth Estimator initialized successfully.")

        print("Press 'q' to quit the video stream.")

        # Create windows
        cv2.namedWindow('Camera Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Depth Map', cv2.WINDOW_AUTOSIZE)

        last_frame_time = time.time()
        fps = 0

        while True:
            current_time = time.time()
            delta_time = current_time - last_frame_time
            last_frame_time = current_time
            if delta_time > 0:
                fps = 1.0 / delta_time

            success, frame_bgr = camera.get_frame()

            if not success:
                print("Failed to capture frame or end of video stream.")
                if camera.camera_id != 0: # If it's a video file, it might have just ended
                    print("End of video file reached.")
                break

            # --- Depth Estimation ---
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Estimate depth
            # For simple viewing, we might not apply a large scale factor, 
            # or we use a small one. The default of 1.0 in estimate_depth is fine for relative viz.
            # If a specific scale was needed for metric interpretation here, it would be passed.
            # For example: depth_map_metric = midas.estimate_depth(frame_rgb, output_scale_factor=10.0)
            depth_map_relative = midas.estimate_depth(frame_rgb) # Uses default output_scale_factor=1.0

            # --- Visualization ---
            # Normalize depth map for visualization (0-255, uint8)
            # This will normalize whatever range depth_map_relative has to 0-255.
            depth_normalized = cv2.normalize(depth_map_relative, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

            # Display FPS on the frames
            cv2.putText(frame_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(depth_colormap, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


            # Display the original frame and the colorized depth map
            cv2.imshow('Camera Feed', frame_bgr)
            cv2.imshow('Depth Map', depth_colormap)

            # Wait for 1ms. If 'q' is pressed, break the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame and depth map
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"output_frame_{timestamp}.png", frame_bgr)
                cv2.imwrite(f"output_depth_{timestamp}.png", depth_colormap)
                # Also save raw depth data if needed (e.g., as a .npy file)
                # np.save(f"output_depth_raw_{timestamp}.npy", depth_map)
                print(f"Saved frame and depth map at {timestamp}")


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
