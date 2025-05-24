from incremental_sfm import run_sfm_system
import os # For a more robust file check

# Optionally, add any other high-level setup or argument parsing here in the future

if __name__ == "__main__":
    print("Starting Real-Time 3D Reconstruction System...")
    
    calibration_file = "calibration_params.npz"
    
    if not os.path.exists(calibration_file):
        print(f"CRITICAL ERROR: Calibration file '{calibration_file}' not found.")
        print("Please run 'calibrate_camera.py' first to generate this file.")
        print("Exiting system.")
        exit()
    else:
        print(f"Calibration file '{calibration_file}' found.")
        print("Ensure this calibration is accurate for your current camera setup.")
        input("Press Enter to continue if calibration is appropriate, or Ctrl+C to exit and re-run calibration...")
            
    run_sfm_system()
    print("System shut down.")
