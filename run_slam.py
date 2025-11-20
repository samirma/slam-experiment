import cv2
import subprocess
import sys
import argparse
import time
import threading

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

def run_slam_process(cam_id):
    print(f"Starting SLAM on Camera {cam_id}...")
    # Command to run main.py
    cmd = [sys.executable, "MASt3R-SLAM/main.py", "--dataset", f"webcam{cam_id}", "--config", "MASt3R-SLAM/config/base.yaml"]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"SLAM process for Camera {cam_id} failed or was stopped.")

def main():
    parser = argparse.ArgumentParser(description="Run SLAM on available cameras.")
    parser.add_argument("--cam-id", type=int, help="Specific camera ID to use. If not set, interactive mode.")
    parser.add_argument("--all", action="store_true", help="Run on all available cameras simultaneously (High Load!)")
    parser.add_argument("--load-map", action="store_true", help="Attempt to load existing map (Not fully supported by MASt3R-SLAM)")

    args = parser.parse_args()

    if args.load_map:
        print("WARNING: Map loading requested, but MASt3R-SLAM does not natively support loading a full persistent map from disk for relocalization.")
        print("A new map will be created. Relocalization is only supported within the session.")

    cameras = list_cameras()

    if not cameras:
        print("No cameras detected.")
        return

    if args.cam_id is not None:
        if args.cam_id in cameras:
            run_slam_process(args.cam_id)
        else:
            print(f"Camera {args.cam_id} not found.")
        return

    if args.all:
        print("WARNING: Running SLAM on multiple cameras simultaneously requires significant GPU resources (e.g. RTX 3090/4090).")
        threads = []
        for cam in cameras:
            t = threading.Thread(target=run_slam_process, args=(cam,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        return

    # Interactive mode
    print("Available cameras:", cameras)
    print("Enter camera ID to use, 'all' to use all, or 'q' to quit.")
    choice = input("Choice: ")

    if choice.lower() == 'q':
        return
    elif choice.lower() == 'all':
        print("WARNING: Running SLAM on multiple cameras simultaneously requires significant GPU resources.")
        threads = []
        for cam in cameras:
            t = threading.Thread(target=run_slam_process, args=(cam,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    else:
        try:
            cam_id = int(choice)
            if cam_id in cameras:
                run_slam_process(cam_id)
            else:
                print("Invalid camera ID.")
        except ValueError:
            print("Invalid input.")

if __name__ == "__main__":
    main()
