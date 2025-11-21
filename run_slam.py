import cv2
import subprocess
import sys
import argparse
import threading
import os

def run_slam_process(dataset_arg, mode="base", port=9876):
    print(f"Starting SLAM on input: {dataset_arg}...")
    # Command to run mast3r_slam_inference.py
    # The new repo uses tools/mast3r_slam_inference.py

    config_file = "MASt3R-SLAM/config/base.yaml"
    if mode == "fast":
        config_file = "MASt3R-SLAM/config/fast.yaml"

    img_size = 512
    if mode == "fast":
        img_size = 224

    cmd = [
        sys.executable, "MASt3R-SLAM/tools/mast3r_slam_inference.py",
        "--dataset", str(dataset_arg),
        "--config", config_file,
        "--img-size", str(img_size),
        "--rerun-port", str(port)
    ]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"SLAM process for {dataset_arg} failed or was stopped.")

def start_gradio_app():
    print("Starting Gradio App...")
    cmd = [sys.executable, "MASt3R-SLAM/tools/gradio-app.py"]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Gradio app failed or was stopped.")


def main():
    parser = argparse.ArgumentParser(description="Run MASt3R-SLAM with Rerun visualization.")
    parser.add_argument("--input", type=str, help="Input video file path or camera index (default: run Gradio app).")
    parser.add_argument("--mode", type=str, choices=["base", "fast"], default="base", help="Mode: 'base' (accurate, 512px) or 'fast' (less accurate, 224px).")
    parser.add_argument("--app", action="store_true", help="Run the Gradio App interface.")
    parser.add_argument("--rerun-port", type=int, default=9876, help="Port for Rerun visualization.")

    args = parser.parse_args()

    # If --app is specified or no input is given, run the Gradio app
    if args.app or not args.input:
        start_gradio_app()
        return

    # Run CLI inference
    if args.input:
        dataset_arg = args.input
        # If input is a digit, assume it's a webcam index, but the script likely expects "webcamX" or just int if handled by simplecv
        # Looking at typical mast3r-slam usage, if it uses simplecv/opencv, integer might work or path.
        # The previous run_slam.py prepended "webcam", but let's pass as is first or check simplecv docs if available.
        # Assuming the underlying script handles paths vs ints.

        run_slam_process(dataset_arg, mode=args.mode, port=args.rerun_port)

if __name__ == "__main__":
    main()
