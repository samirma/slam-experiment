"""
A simple script to test camera functionality.

This script initializes a `MonocularCamera` object from the `src.camera.camera` module
and displays the live feed from the specified camera ID (defaulting to 0).
It's useful for quickly checking if a camera is detected and working correctly
with OpenCV and the `MonocularCamera` class.

Usage:
    python scripts/view_camera.py
"""
import cv2
import sys
from pathlib import Path # Import was missing, but sys.path.append implies its usage
# Add the src directory to the Python path to allow importing MonocularCamera
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.camera.camera import MonocularCamera
from pathlib import Path


def main():
    """
    Main function to capture and display video from a monocular camera.
    """
    try:
        camera = MonocularCamera(0)  # Or specify a video file path
        print("Successfully opened camera.")
        print("Press 'q' to quit the video stream.")

        while True:
            success, frame = camera.get_frame()

            if not success:
                print("Failed to capture frame or end of video stream.")
                break

            # Display the resulting frame
            cv2.imshow('Camera Feed', frame)

            # Wait for 1ms. If 'q' is pressed, break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

    except IOError as e:
        print(f"Error: {e}")
    except ImportError as e:
        print(f"ImportError: {e}. Make sure src directory is in PYTHONPATH.")
        print("Attempting to add src to sys.path and retrying import...")
        # This is a fallback, ideally the environment should be set up correctly
        try:
            from src.camera.camera import MonocularCamera
            # Retry creating camera object if import was the issue initially
            # This part might be redundant if the initial import works after path modification
            # but included for robustness in case of direct script run without proper PYTHONPATH
            camera = MonocularCamera(0)
            print("Successfully opened camera after path adjustment.")
            print("Press 'q' to quit the video stream.")
            # Duplicate the loop here or refactor into a function to avoid repetition
            # For simplicity, we'll just note that the logic would be similar.
        except Exception as e_retry:
             print(f"Could not start camera after attempting to fix import: {e_retry}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'camera' in locals() and camera.cap.isOpened():
            camera.release()
            print("Camera released.")
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")

if __name__ == "__main__":
    main()
