import os
import sys
import subprocess
import shutil

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

def main():
    print("Checking environment...")

    # 1. Check CUDA
    if not check_command("nvcc"):
        print("WARNING: nvcc (CUDA) not found. MASt3R-SLAM requires CUDA.")
        print("Please ensure CUDA is installed and in your PATH.")

    # Install generic dependencies
    print("Installing Python dependencies...")
    # List from pixi.toml and pyproject.toml
    # Note: eigen is a C++ library, not installed via pip here (assumed system install)
    dependencies = [
        "numpy<2",
        "tyro>=0.9.1,<0.10",
        "rerun-sdk>=0.22.0,<0.23",
        "beartype>=0.20.0,<0.21",
        "jaxtyping>=0.2.36,<0.3",
        "gradio<5",
        "opencv-python>=4.11.0,<5",
        "pyserde>=0.23.0,<0.24",
        "open3d>=0.19.0,<0.20",
        "einops",
        "pyrealsense2",
        "evo",
        "natsort",
        "pykdtree",
        # From thirdparty/mast3r/setup.py
        "scikit-learn",
        "roma",
        "matplotlib",
        "tqdm",
        "scipy",
        "trimesh",
        "tensorboard",
        "pyglet",
        "huggingface_hub[torch]>=0.22",
    ]

    cmd = f"{sys.executable} -m pip install " + " ".join([f"'{d}'" for d in dependencies])
    if not run_command(cmd):
        sys.exit(1)

    # Install simplecv
    print("Installing simplecv...")
    if not run_command(f"{sys.executable} -m pip install git+https://github.com/pablovela5620/simplecv.git"):
        sys.exit(1)

    # Install gradio-rerun wheel
    print("Installing gradio-rerun...")
    if not run_command(f"{sys.executable} -m pip install https://huggingface.co/datasets/pablovela5620/gradio-rr-wheels/resolve/main/gradio_rerun-0.0.11-py3-none-any.whl"):
        sys.exit(1)

    # Install lietorch
    print("Installing lietorch...")
    # Try wheel first (from pixi.toml)
    # wheel_url = "https://huggingface.co/datasets/pablovela5620/mast3r-slam-whls/resolve/main/lietorch-0.2-cp311-cp311-linux_x86_64.whl"
    # if not run_command(f"{sys.executable} -m pip install {wheel_url}"):
    #     print("Wheel installation failed, trying from source...")
    if not run_command(f"{sys.executable} -m pip install git+https://github.com/princeton-vl/lietorch.git"):
        sys.exit(1)


    # Install thirdparty/mast3r
    # This handles curope and asmk compilation
    print("Installing mast3r (including curope and asmk)...")
    if os.path.exists("MASt3R-SLAM/thirdparty/mast3r"):
         if not run_command(f"{sys.executable} -m pip install -e MASt3R-SLAM/thirdparty/mast3r"):
             print("ERROR: Failed to install thirdparty/mast3r")
             sys.exit(1)
    else:
        print("ERROR: MASt3R-SLAM/thirdparty/mast3r not found")
        sys.exit(1)

    # Download checkpoints
    print("Downloading checkpoints...")
    checkpoints_dir = "MASt3R-SLAM/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    # We can use huggingface-cli if installed, which is installed via dependencies
    try:
        # Check if files exist to avoid re-downloading via CLI if possible, though CLI handles it too.
        # Using python script to download to ensure cross-platform compatibility if needed,
        # but simple CLI command is easier if we trust huggingface_hub is installed.
        files = [
            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        ]
        missing = [f for f in files if not os.path.exists(os.path.join(checkpoints_dir, f))]

        if missing:
            run_command(f"huggingface-cli download pablovela5620/mast3r-slam {' '.join(files)} --repo-type model --local-dir {checkpoints_dir}")
    except Exception as e:
        print(f"Error downloading checkpoints: {e}")

    # Install MASt3R-SLAM
    print("Installing MASt3R-SLAM...")
    if not run_command(f"{sys.executable} -m pip install --no-build-isolation -e MASt3R-SLAM"):
        sys.exit(1)

    print("Setup complete.")

if __name__ == "__main__":
    main()
