import os
import sys
import subprocess
import shutil
import requests
import glob

def check_command(command):
    return shutil.which(command) is not None

def run_command(command, cwd=None):
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        return False
    return True

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists, skipping download.")
        return
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        print(f"Failed to download {url}")

def verify_calibration():
    print("Verifying calibration files...")
    # Check for standard calibration files
    if os.path.exists("calibration.yaml") or os.path.exists("intrinsics.yaml"):
        print("Calibration file found.")
        return True

    print("WARNING: No calibration file found (calibration.yaml or intrinsics.yaml).")
    print("MASt3R-SLAM can run without calibration, but accuracy might be lower.")
    print("To calibrate your cameras, you can use OpenCV's calibration sample:")
    print("https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html")
    return False

def apply_patches():
    print("Applying patches...")
    # Ensure submodule is initialized
    if not os.path.exists("MASt3R-SLAM/main.py"):
        print("MASt3R-SLAM submodule not found. Initializing...")
        run_command("git submodule update --init --recursive")

    # Copy modified dataloader
    if os.path.exists("patches/dataloader.py"):
        dest = "MASt3R-SLAM/mast3r_slam/dataloader.py"
        print(f"Copying patches/dataloader.py to {dest}")
        shutil.copy("patches/dataloader.py", dest)
    else:
        print("ERROR: patches/dataloader.py not found!")

def main():
    print("Checking environment...")

    # 1. Check CUDA
    if not check_command("nvcc"):
        print("WARNING: nvcc (CUDA) not found. MASt3R-SLAM requires CUDA.")
        print("Please ensure CUDA is installed and in your PATH.")

    # 2. Apply Code Patches
    apply_patches()

    # 3. Check Calibration
    verify_calibration()

    # 4. Setup Checkpoints
    print("Setting up checkpoints...")
    os.makedirs("MASt3R-SLAM/checkpoints", exist_ok=True)

    base_url = "https://download.europe.naverlabs.com/ComputerVision/MASt3R"
    files = [
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
    ]

    for f in files:
        download_file(f"{base_url}/{f}", f"MASt3R-SLAM/checkpoints/{f}")

    # 5. Install custom dependencies (if not already)
    # Try to install lietorch
    if not run_command(f"{sys.executable} -c 'import lietorch'"):
         print("lietorch not found. Attempting to install...")
         if not os.path.exists("lietorch"):
             run_command("git clone https://github.com/princeton-vl/lietorch.git")
         # Try to install
         run_command(f"cd lietorch && {sys.executable} setup.py install")

    # Install main requirements from submodule if present
    if os.path.exists("MASt3R-SLAM"):
        print("Installing MASt3R-SLAM dependencies...")
        run_command(f"cd MASt3R-SLAM && {sys.executable} -m pip install -e .")

    print("Setup complete (or attempted).")
    print("To run SLAM: python run_slam.py")

if __name__ == "__main__":
    main()
