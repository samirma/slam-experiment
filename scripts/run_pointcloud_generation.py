"""
Main script for the 3D SLAM and Reconstruction System.

This script integrates all components of the system:
1.  Camera Input: Captures live video frames using `MonocularCamera`.
2.  SLAM Frontend: Estimates camera pose and generates a sparse map using `SLAMFrontend`.
3.  Depth Estimation: Estimates depth from each frame using `MiDaSDepthEstimator`.
4.  3D Reconstruction: Builds a dense 3D map of the environment using `PointCloudMapper`
    with TSDF (Truncated Signed Distance Function) integration, guided by SLAM poses.

The script initializes these modules, runs a main loop to process frames,
and visualizes the generated dense and sparse point clouds in real-time using Open3D.
It supports loading an initial map (for TSDF visualization), saving the reconstructed
dense map, and extracting/viewing the 3D mesh.

Key Features:
- Real-time monocular SLAM pipeline with `SLAMFrontend`.
- TSDF-based dense reconstruction for robust 3D mapping.
- Visualization of the live dense point cloud (from TSDF).
- Visualization of the live sparse map points (from SLAM).
- Option to save the reconstructed dense map (point cloud) and mesh.
- Option to load a pre-existing point cloud map for visualization (TSDF only).
- Camera calibration support (uses `data/camera_calibration.yaml`).

Keyboard Controls (in Open3D visualizer window):
  - 'q': Quit the application.
  - 'k': Save the current dense map (TSDF-extracted point cloud) to `data/generated_map_tsdf.ply`.
  - 'l': Load a dense map from `data/generated_map_tsdf.ply` (for TSDF visualization, resets live TSDF).
  - 'm': Extract and view the mesh from the current TSDF volume in a new window.
"""
import cv2
import numpy as np
import sys
from pathlib import Path
import open3d as o3d
import time
import os
import argparse

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.camera.camera import MonocularCamera
from src.depth_estimation import MiDaSDepthEstimator # Uses __init__.py for MiDaS
from src.slam import SLAMFrontend # <<<< CHANGED FROM VisualOdometry
from src.utils import CameraParams # Uses __init__.py for CameraParams
from src.reconstruction import PointCloudMapper # Uses __init__.py for PointCloudMapper

def main():
    """
    Main function to initialize and run the full 3D reconstruction pipeline.
    Handles camera capture, SLAM, depth estimation, TSDF mapping,
    and visualization of both dense and sparse maps.
    """
    parser = argparse.ArgumentParser(
        description="Run the 3D SLAM and Reconstruction pipeline with SLAMFrontend and TSDF integration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--load_map", 
        type=str, 
        help="Path to a .ply point cloud file to load for initial TSDF visualization. "
             "Note: Live TSDF processing will start with a fresh volume regardless of this loaded map."
    )
    args = parser.parse_args()

    print("Starting Full 3D SLAM and Reconstruction Pipeline with SLAMFrontend...")
    camera = None
    vis = None
    # Reference for the dense point cloud geometry (TSDF)
    dense_pcd_vis_geom = None 
    # Reference for the sparse map geometry (SLAM)
    sparse_map_o3d_geom = None

    # --- Configuration ---
    IMAGE_WIDTH = 640 
    IMAGE_HEIGHT = 480
    CALIBRATION_FILE = "data/camera_calibration.yaml"
    
# --- IMPORTANT: CALIBRATION REQUIRED for DEPTH_OUTPUT_SCALE_FACTOR ---
# The SLAM system (`SLAMFrontend`) now establishes its own internal (arbitrary but consistent)
# scale for camera poses and the sparse map, based on `initial_baseline_scale` used
# during the triangulation of the first two keyframes.
#
# MiDaS produces relative depth maps. To achieve a coherent 3D reconstruction (where the
# dense map from MiDaS aligns with the SLAM's sparse map and camera trajectory),
# you MUST calibrate `DEPTH_OUTPUT_SCALE_FACTOR`. This factor scales the raw MiDaS output.
#
# The goal of this calibration is to make the scaled MiDaS depth compatible with the
# SLAM system's internal scale.
#
# Calibration Steps (Iterative and Visual):
# 1. Initial Guess for `DEPTH_OUTPUT_SCALE_FACTOR`:
#    You can start with a rough guess. A previous method was:
#    a. Point camera at an object of known true distance (e.g., `d_true` meters).
#    b. Temporarily modify `src/depth_estimation/midas.py` (in `estimate_depth`, before scaling)
#       to print the raw MiDaS output for that object (e.g., `d_midas_raw`).
#       `print(f"Raw MiDaS depth for center: {depth_map[h//2, w//2]}")` (remove after).
#    c. Initial `DEPTH_OUTPUT_SCALE_FACTOR = d_true / d_midas_raw`.
#    This gives a starting point, but it likely needs refinement.
#
# 2. Run the System: Execute this `run_pointcloud_generation.py` script.
#    Observe the Open3D window. You should eventually see:
#       - The camera trajectory (from SLAM).
#       - Sparse map points (from SLAM, if visualization is enabled).
#       - The dense point cloud (from MiDaS depth integrated by PointCloudMapper).
#
# 3. Visual Tuning:
#    - Adjust `DEPTH_OUTPUT_SCALE_FACTOR` in this script.
#    - Re-run the script.
#    - Observe the dense point cloud.
#      - If the dense cloud appears "too flat" or "too close" to the camera relative
#        to the camera's movement path (from SLAM), your `DEPTH_OUTPUT_SCALE_FACTOR` might be too small. Try increasing it.
#      - If the dense cloud appears "too deep," "too far," or "stretched out" relative
#        to camera movement, your factor might be too large. Try decreasing it.
#    - The goal is to find a factor where the dense reconstruction looks geometrically
#      consistent with the camera motion and the sparse SLAM map. For example, if the
#      camera moves forward, the dense cloud should extend forward coherently.
#
# 4. Iteration: Repeat step 3 until the reconstruction appears reasonable.
#    This factor is sensitive and scene-dependent. A good value for one environment
#    might need adjustment for another.
#
# Example (replace with your calibrated and tuned value):
# DEPTH_OUTPUT_SCALE_FACTOR = 10.0 # Default, NEEDS CALIBRATION & TUNING
    DEPTH_OUTPUT_SCALE_FACTOR = 10.0 
    DEFAULT_SAVE_PATH = "data/generated_map_tsdf.ply" 
    DEFAULT_LOAD_PATH = "data/generated_map_tsdf.ply" 

    # --- TSDF Parameters for PointCloudMapper ---
    TSDF_VOXEL_LENGTH = 0.02  
    TSDF_SDF_TRUNC = 0.04    

    try:
        # --- Initialization ---
        print("Initializing camera...")
        camera = MonocularCamera(0) 

        print("Initializing Camera Parameters...")
        cam_params = CameraParams(calibration_file_path=CALIBRATION_FILE,
                                  default_image_width=IMAGE_WIDTH,
                                  default_image_height=IMAGE_HEIGHT)
        IMAGE_WIDTH, IMAGE_HEIGHT = cam_params.get_image_dimensions()
        print(f"Camera parameters initialized using '{CALIBRATION_FILE}'. Loaded K=\n{cam_params.get_K()}")
        print(f"Image dimensions from CameraParams: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")

        print("Initializing MiDaS Depth Estimator...")
        depth_estimator = MiDaSDepthEstimator()

        print("Initializing SLAM Frontend...") # <<<< CHANGED
        slam = SLAMFrontend(camera_params=cam_params) # <<<< CHANGED

        print("Initializing PointCloud Mapper with TSDF...")
        pointcloud_mapper = PointCloudMapper(camera_params=cam_params, 
                                             voxel_length_tsdf=TSDF_VOXEL_LENGTH, 
                                             sdf_trunc_tsdf=TSDF_SDF_TRUNC)
        pointcloud_mapper._loaded_external_pcd = None

        print("Initializing Open3D Visualizer...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="SLAM + Dense Reconstruction", width=1280, height=720) # Wider window
        vis.get_render_option().point_size = 2.0 # Default point size
        vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1]) 
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame) 

        # Geometry for dense point cloud (TSDF)
        dense_pcd_vis_geom = o3d.geometry.PointCloud()
        vis.add_geometry(dense_pcd_vis_geom)

        # Geometry for sparse map points (SLAM)
        sparse_map_o3d_geom = o3d.geometry.PointCloud()
        vis.add_geometry(sparse_map_o3d_geom)

        if args.load_map:
            print(f"Attempting to load map from command line argument: {args.load_map}")
            if pointcloud_mapper.load_map(args.load_map) and pointcloud_mapper._loaded_external_pcd:
                # This loaded map is for the DENSE visualization (TSDF)
                dense_pcd_vis_geom.points = pointcloud_mapper._loaded_external_pcd.points
                dense_pcd_vis_geom.colors = pointcloud_mapper._loaded_external_pcd.colors
                if dense_pcd_vis_geom.has_points():
                    vis.update_geometry(dense_pcd_vis_geom)
                    vis.reset_view_point(True)
                    print(f"Map from {args.load_map} loaded and added to dense visualizer.")
                else:
                    print(f"Loaded map from {args.load_map} is empty.")
            else:
                print(f"Failed to load map from {args.load_map}.")
        else:
            print("No initial map loaded for TSDF. Live reconstruction will populate it.")

        last_time = time.time()
        print("Press 'q' in the OpenCV window or Open3D window to quit.")
        print(f"Press 'k' to save the current TSDF-extracted dense map to {DEFAULT_SAVE_PATH}.")
        print(f"Press 'l' to load a PLY map from {DEFAULT_LOAD_PATH} (for TSDF visualization, TSDF will reset).")
        print("Press 'm' to extract and view the mesh from TSDF.")

        # --- Main Loop ---
        while True:
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time

            success, frame_bgr = camera.get_frame()
            if not success:
                print("Failed to capture frame or end of video stream.")
                break
            
            if frame_bgr.shape[1] != IMAGE_WIDTH or frame_bgr.shape[0] != IMAGE_HEIGHT:
                 frame_bgr = cv2.resize(frame_bgr, (IMAGE_WIDTH, IMAGE_HEIGHT))

            # --- SLAM Frontend ---
            R_w2c, t_w2c, slam_kps, slam_matches = slam.process_frame(frame_bgr.copy()) # Pass a copy for SLAM

            # --- Depth Estimation ---
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            depth_map_m = depth_estimator.estimate_depth(frame_rgb, output_scale_factor=DEPTH_OUTPUT_SCALE_FACTOR)
            depth_map_m = np.clip(depth_map_m, 0.01, 30.0) 

            # --- Dense Point Cloud Generation and Mapping (TSDF) ---
            # R_w2c, t_w2c is from SLAMFrontend
            pointcloud_mapper.integrate_frame_tsdf(frame_bgr, depth_map_m, R_w2c, t_w2c)
            
            # --- Visualization Update ---
            # Update Dense Cloud (TSDF)
            current_dense_pcd_from_tsdf = pointcloud_mapper.get_global_point_cloud()
            if current_dense_pcd_from_tsdf.has_points():
                dense_pcd_vis_geom.points = current_dense_pcd_from_tsdf.points
                dense_pcd_vis_geom.colors = current_dense_pcd_from_tsdf.colors
                vis.update_geometry(dense_pcd_vis_geom)
            else:
                dense_pcd_vis_geom.clear()
                vis.update_geometry(dense_pcd_vis_geom)

            # Update Sparse Map (SLAM)
            if slam.map_points is not None and slam.map_points.shape[0] > 0:
                sparse_map_o3d_geom.points = o3d.utility.Vector3dVector(slam.map_points)
                # Color sparse map points (e.g., green)
                num_sparse_points = slam.map_points.shape[0]
                sparse_map_colors = np.tile(np.array([0.0, 1.0, 0.0]), (num_sparse_points, 1))
                sparse_map_o3d_geom.colors = o3d.utility.Vector3dVector(sparse_map_colors)
                vis.update_geometry(sparse_map_o3d_geom)
            else:
                sparse_map_o3d_geom.clear()
                vis.update_geometry(sparse_map_o3d_geom)

            if not vis.poll_events(): 
                print("Open3D window closed by user. Exiting loop.")
                break
            vis.update_renderer()
            
            # --- OpenCV Displays ---
            # Display camera feed with SLAM keypoints
            frame_with_slam_kps = cv2.drawKeypoints(frame_bgr, slam_kps, None, color=(0,255,0))
            # You could also draw slam_matches if they are relevant for the current SLAM state
            # For example, if slam.state == "TRACKING_WITH_LOCAL_MAP", slam_matches are matches to map points
            # If slam.state == "WAITING_FOR_SECOND_FRAME", slam_matches are between KF1 and current frame
            cv2.putText(frame_with_slam_kps, f"FPS: {fps:.2f} SLAM State: {slam.state}", (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow('Camera Feed & SLAM Keypoints', frame_with_slam_kps)

            depth_display = cv2.normalize(depth_map_m, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)
            cv2.imshow('Depth Map Display', depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting via OpenCV window...")
                break
            elif key == ord('k'):
                print(f"Saving TSDF dense map to {DEFAULT_SAVE_PATH}...")
                pointcloud_mapper.save_map(DEFAULT_SAVE_PATH)
            elif key == ord('l'):
                print(f"Attempting to load PLY map from {DEFAULT_LOAD_PATH} for TSDF visualization...")
                print("Resetting internal TSDF volume.")
                pointcloud_mapper.tsdf_volume.reset() 
                if dense_pcd_vis_geom: # Should exist
                    dense_pcd_vis_geom.clear()
                    vis.update_geometry(dense_pcd_vis_geom) 

                if pointcloud_mapper.load_map(DEFAULT_LOAD_PATH) and pointcloud_mapper._loaded_external_pcd:
                    dense_pcd_vis_geom.points = pointcloud_mapper._loaded_external_pcd.points
                    dense_pcd_vis_geom.colors = pointcloud_mapper._loaded_external_pcd.colors
                    if dense_pcd_vis_geom.has_points():
                        vis.update_geometry(dense_pcd_vis_geom)
                        vis.reset_view_point(True) 
                        print("Loaded PLY map displayed in TSDF visualizer. SLAM state is not reset.")
                    else:
                        print("Loaded PLY map is empty.")
                else:
                    print(f"Failed to load PLY map from {DEFAULT_LOAD_PATH}.")
            elif key == ord('m'): 
                print("Extracting mesh from TSDF volume...")
                mesh = pointcloud_mapper.get_mesh()
                if mesh.has_vertices() and mesh.has_triangles():
                    print(f"Mesh extracted with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
                    o3d.visualization.draw_geometries([mesh], window_name="Extracted Mesh")
                else:
                    print("Could not extract a valid mesh from the TSDF volume.")

    except Exception as e:
        print(f"A critical error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up resources...")
        if camera is not None and camera.cap.isOpened():
            camera.release()
            print("Camera released.")
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")
        if vis is not None:
            vis.destroy_window()
            print("Open3D visualizer destroyed.")
        print("Application finished.")

if __name__ == "__main__":
    main()
