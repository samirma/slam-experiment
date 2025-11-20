import cv2
import subprocess
import sys
import argparse
import time
import threading
import os

def list_cameras():
    print("Detecting cameras...")
    available_cameras = []
    # Check first 10 indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is available")
            available_cameras.append(i)
            cap.release()
    return available_cameras

def run_slam_process(dataset_arg):
    print(f"Starting SLAM on input: {dataset_arg}...")
    # Command to run main.py
    cmd = [sys.executable, "MASt3R-SLAM/main.py", "--dataset", str(dataset_arg), "--config", "MASt3R-SLAM/config/base.yaml"]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"SLAM process for {dataset_arg} failed or was stopped.")

def main():
    parser = argparse.ArgumentParser(description="Run SLAM on available cameras or video files.")
    parser.add_argument("--cam-id", type=int, help="Specific camera ID to use (deprecated, use --inputs).")
    parser.add_argument("--inputs", nargs='+', help="List of inputs to process. Can be camera IDs (0, 1) or file paths (video.mp4).")
    parser.add_argument("--all", action="store_true", help="Run on all detected webcams simultaneously (High Load!)")
    parser.add_argument("--load-map", action="store_true", help="Attempt to load existing map (Not fully supported by MASt3R-SLAM)")

    args = parser.parse_args()

    if args.load_map:
        print("WARNING: Map loading requested, but MASt3R-SLAM does not natively support loading a full persistent map from disk for relocalization.")
        print("A new map will be created. Relocalization is only supported within the session.")

    # Handle --inputs (Explicit list)
    if args.inputs:
        if len(args.inputs) > 1:
             print("WARNING: Running SLAM on multiple inputs simultaneously requires significant GPU resources.")

        threads = []
        for inp in args.inputs:
            # Determine if input is an ID or path
            dataset_arg = inp
            if inp.isdigit():
                 # Assume it's a camera ID
                 dataset_arg = f"webcam{inp}"
            # If it's a file or "webcamX", pass as is.

            t = threading.Thread(target=run_slam_process, args=(dataset_arg,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        return

    # Handle --cam-id (Legacy single cam)
    if args.cam_id is not None:
        run_slam_process(f"webcam{args.cam_id}")
        return

    # Handle --all (Auto-detect)
    cameras = list_cameras()
    if args.all:
        if not cameras:
            print("No cameras detected.")
            return

        print("WARNING: Running SLAM on multiple cameras simultaneously requires significant GPU resources (e.g. RTX 3090/4090).")
        threads = []
        for cam in cameras:
            t = threading.Thread(target=run_slam_process, args=(f"webcam{cam}",))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        return

    # Interactive mode
    if not cameras:
        print("No cameras detected.")
        # We can still ask for file input
    else:
        print("Available cameras:", cameras)

    print("Enter camera ID (e.g. 0), file path, list separated by space, 'all', or 'q' to quit.")
    choice = input("Choice: ")

    if choice.lower() == 'q':
        return
    elif choice.lower() == 'all':
        if not cameras:
             print("No cameras to run 'all' on.")
             return
        print("WARNING: Running SLAM on multiple cameras simultaneously requires significant GPU resources.")
        threads = []
        for cam in cameras:
            t = threading.Thread(target=run_slam_process, args=(f"webcam{cam}",))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    else:
        # Split by space to support multiple inputs in interactive mode too
        inputs = choice.split()
        threads = []
        for inp in inputs:
            dataset_arg = inp
            if inp.isdigit():
                dataset_arg = f"webcam{inp}"

            t = threading.Thread(target=run_slam_process, args=(dataset_arg,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

if __name__ == "__main__":
    main()
