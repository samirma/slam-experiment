# MASt3R-SLAM with Rerun Visualization

This repository hosts a Dockerized version of [MASt3R-SLAM](https://github.com/rerun-io/mast3r-slam), which is an unofficial implementation of "MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors" using Rerun for visualization.

## Setup

The environment is set up using Docker.

1.  **Build the Docker image:**

    ```bash
    docker build -t mast3r-slam .
    ```

    This will install all dependencies, including CUDA 12.4, Python 3.11, PyTorch, and compile the necessary custom kernels (`curope`, `asmk`). It will also download the required model checkpoints.

## Usage

### Running the Gradio App

To run the interactive Gradio interface:

```bash
docker run --gpus all -p 7860:7860 -p 9876:9876 -it mast3r-slam python3 run_slam.py --app
```

Then open `http://localhost:7860` in your browser.

### Running on Video File or Camera

To run SLAM on a specific video file:

```bash
docker run --gpus all -p 9876:9876 -v /path/to/your/data:/data -it mast3r-slam python3 run_slam.py --input /data/your_video.mp4
```

To run on a webcam (pass device to docker):

```bash
docker run --gpus all -p 9876:9876 --device /dev/video0 -it mast3r-slam python3 run_slam.py --input 0
```

### Real-time Visualization

The application starts a Rerun server on port 9876. You can connect to it using a Rerun viewer on your host machine.

1.  Install Rerun on your host: `pip install rerun-sdk`
2.  Run `rerun` on your host.
3.  Or, if the script logs to a file (rrd), you can open that. By default, `mast3r-slam` with rerun usually streams to the connected viewer or starts a web viewer.
    *   If running Gradio, the visualization is embedded.
    *   If running CLI, ensure you have port 9876 forwarded.

### Modes

You can choose between `base` (default, more accurate, 512px) and `fast` (less accurate, 224px) modes using `--mode`:

```bash
python3 run_slam.py --input ... --mode fast
```

## Acknowledgements

*   [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM)
*   [rerun-io/mast3r-slam](https://github.com/rerun-io/mast3r-slam)
*   [MASt3R](https://github.com/naver/mast3r)
*   [DUSt3R](https://github.com/naver/dust3r)
