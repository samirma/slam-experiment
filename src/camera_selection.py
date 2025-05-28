# src/camera_selection.py
import os
import cv2

CALIBRATION_IMAGE_DIR = "calibration_data"

def get_available_cameras_info(max_cameras_to_check=5):
    """
    Detects available cameras and checks their calibration status.
    
    Returns:
        list: A list of dictionaries, where each dictionary contains:
              'id': camera index (int)
              'description': descriptive string (str)
              'has_calibration': boolean
    """
    available_cameras = []
    print(f"Probing for cameras up to index {max_cameras_to_check-1}...")
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        # Some camera backends (like MSMF) might need a moment to open
        # or might report isOpened() true even if not fully functional.
        # A quick read can be a more robust check for some backends.
        # However, for simplicity and speed, we'll rely on isOpened() for now.
        if cap.isOpened():
            calib_file_path = os.path.join(CALIBRATION_IMAGE_DIR, f"camera_params_idx{i}.npz")
            has_calibration = os.path.exists(calib_file_path)
            
            # Creating a generic description. More specific names are highly platform/backend dependent.
            description = f"Camera ID: {i} (Calibration: {'Yes' if has_calibration else 'No'})"
            
            available_cameras.append({
                'id': i,
                'description': description,
                'has_calibration': has_calibration
            })
            cap.release()
    return available_cameras

def select_camera_interactive():
    """
    Lists available cameras with their calibration status and prompts the user for selection.
    
    Returns:
        int: The selected camera index, or None if no camera is selected or found.
    """
    available_cameras_info = get_available_cameras_info()

    if not available_cameras_info:
        print("No cameras detected or accessible.")
        return None

    print("\nAvailable Cameras:")
    for cam_info in available_cameras_info:
        # Print the description which already includes ID and calibration status
        print(f"- {cam_info['description']}")
    
    valid_ids = [info['id'] for info in available_cameras_info]

    while True:
        try:
            # Using f-string for a cleaner prompt showing available options.
            raw_input_str = input(f"Enter the ID of the camera you want to use (options: {valid_ids}): ")
            selected_id = int(raw_input_str)
            if selected_id in valid_ids:
                print(f"Selected camera ID: {selected_id}")
                return selected_id
            else:
                print(f"Invalid ID. Please choose from {valid_ids}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            # Catching other potential errors during input, though less common here.
            print(f"An unexpected error occurred during input: {e}")
            return None # Exit if something unexpected happens

if __name__ == '__main__':
    print("Attempting to select a camera interactively...")
    selected_camera_index = select_camera_interactive()
    
    if selected_camera_index is not None:
        print(f"Script finished: Camera selected with index: {selected_camera_index}")
    else:
        print("Script finished: No camera was selected or an error occurred during selection.")
