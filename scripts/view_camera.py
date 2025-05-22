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
# Change to module-based import
import src.camera.camera as camera_module
print(f"DEBUG: camera_module imported: {camera_module}")
if hasattr(camera_module, 'MonocularCamera'):
    print(f"DEBUG: MonocularCamera class via module: {camera_module.MonocularCamera}")
else:
    print("DEBUG: MonocularCamera class NOT found in camera_module after import.")
from pathlib import Path
import traceback # For detailed error reporting


def main():
    """
    Main function to capture and display video from a monocular camera.
    """
    camera_instance = None # Define to ensure it's in scope for finally
    try:
        print("DEBUG: Attempting to instantiate MonocularCamera.")
        # Ensure MonocularCamera is treated as the class from the import
        # This line assumes MonocularCamera is correctly imported and is a class
        camera_instance = camera_module.MonocularCamera(0) 
        print("Successfully opened camera.")
        print("Press 'q' to quit the video stream.")

        while True:
            success, frame = camera_instance.get_frame()

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
    except ImportError as e: # This specifically catches if the 'from src.camera.camera import MonocularCamera' fails
        print(f"ImportError: {e}. Make sure src directory is in PYTHONPATH.")
        print("DEBUG: Attempting to re-import module and access MonocularCamera...")
        # This is a fallback, ideally the environment should be set up correctly
        try:
            # Try to re-import locally if the global one failed or this block is hit for other reasons
            import src.camera.camera as camera_module_retry # Use a different alias or re-import
            print(f"DEBUG: camera_module_retry imported: {camera_module_retry}")
            if hasattr(camera_module_retry, 'MonocularCamera'):
                 print(f"DEBUG: MonocularCamera via camera_module_retry: {camera_module_retry.MonocularCamera}")
                 camera_instance = camera_module_retry.MonocularCamera(0) # Assign to camera_instance
                 print("Successfully opened camera after path adjustment and re-import.")
                 print("Press 'q' to quit the video stream.")
                 # Note: The main loop would need to be re-entered or duplicated here for full functionality after re-import.
                 if camera_instance and hasattr(camera_instance, 'cap') and camera_instance.cap.isOpened():
                     print("DEBUG: Camera re-opened, entering dummy loop for cleanup test.")
            else:
                print("DEBUG: MonocularCamera class NOT found in camera_module_retry.")
        except Exception as e_retry:
             print(f"Could not start camera after attempting to fix import: {repr(e_retry)}")
    except NameError as ne:
        print(f"DEBUG: A NameError occurred: {repr(ne)}")
        # Adjusted debug prints
        current_scope_camera_module = None
        if 'camera_module' in globals(): current_scope_camera_module = globals()['camera_module']
        elif 'camera_module' in locals(): current_scope_camera_module = locals()['camera_module'] # Less likely for module
        
        print(f"DEBUG: Is camera_module defined? {current_scope_camera_module is not None}")
        if current_scope_camera_module is not None:
            print(f"DEBUG: Is MonocularCamera in camera_module? {hasattr(current_scope_camera_module, 'MonocularCamera')}")
        traceback.print_exc()
    except UnboundLocalError as ule:
        print(f"DEBUG: An UnboundLocalError occurred: {repr(ule)}")
        # Adjusted debug prints
        current_scope_camera_module = None
        if 'camera_module' in globals(): current_scope_camera_module = globals()['camera_module']
        elif 'camera_module' in locals(): current_scope_camera_module = locals()['camera_module']

        print(f"DEBUG: Is camera_module defined? {current_scope_camera_module is not None}")
        if current_scope_camera_module is not None:
            print(f"DEBUG: Is MonocularCamera in camera_module? {hasattr(current_scope_camera_module, 'MonocularCamera')}")
        traceback.print_exc()
    except Exception as e: # Generic handler
        print(f"An unexpected error occurred in main try: {repr(e)}")
        traceback.print_exc()
    finally:
        if camera_instance is not None and hasattr(camera_instance, 'cap') and camera_instance.cap.isOpened():
            camera_instance.release()
            print("Camera released.")
        else:
            print("DEBUG: Camera not released in finally (either None, or no cap, or not open).")
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")

if __name__ == "__main__":
    main()
