"""
Main script for the 3D SLAM and Reconstruction System.

This script integrates all components of the system:
1.  Camera Input: Captures live video frames using `MonocularCamera`.
2.  Depth Estimation: Estimates depth from each frame using `MiDaSDepthEstimator`.
3.  Visual Odometry: Tracks camera pose using `VisualOdometry`.
4.  3D Reconstruction: Builds a 3D map of the environment using `PointCloudMapper`
    with TSDF (Truncated Signed Distance Function) integration.

The script initializes these modules, runs a main loop to process frames,
and visualizes the generated 3D point cloud in real-time using Open3D.
It supports loading an initial map, saving the reconstructed map, and
extracting/viewing the 3D mesh.

Key Features:
- Real-time monocular SLAM pipeline.
- TSDF-based dense reconstruction for robust 3D mapping.
- Visualization of the live point cloud.
- Option to save the reconstructed map (point cloud) and mesh.
- Option to load a pre-existing point cloud map for visualization.
- Camera calibration support (uses `data/camera_calibration.yaml`).

Keyboard Controls (in Open3D visualizer window):
  - 'q': Quit the application.
  - 'k': Save the current map (TSDF-extracted point cloud) to `data/generated_map_tsdf.ply`.
  - 'l': Load a map from `data/generated_map_tsdf.ply` (for visualization, resets live TSDF).
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
from src.slam import VisualOdometry # Uses __init__.py for VO
from src.utils import CameraParams # Uses __init__.py for CameraParams
from src.reconstruction import PointCloudMapper # Uses __init__.py for PointCloudMapper

def main():
    """
    Main function to initialize and run the full 3D reconstruction pipeline.
    Handles camera capture, depth estimation, visual odometry, TSDF mapping,
    and visualization.
    """
    parser = argparse.ArgumentParser(
        description="Run the 3D SLAM and Reconstruction pipeline using TSDF integration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument(
        "--load_map", 
        type=str, 
        help="Path to a .ply point cloud file to load for initial visualization. "
             "Note: Live TSDF processing will start with a fresh volume regardless of this loaded map."
    )
    args = parser.parse_args()

    print("Starting Full 3D SLAM and Reconstruction Pipeline...")
    camera = None
    vis = None
    # Keep a reference to the point cloud geometry in the visualizer for removal/update
    pcd_vis_geom = None 

    # --- Configuration ---
    IMAGE_WIDTH = 640 # Default, will be updated by CameraParams if calibration file is loaded
    IMAGE_HEIGHT = 480 # Default
    # FX = 550.0  # Placeholder, use calibrated values - Now from CameraParams
    # FY = 550.0  # Placeholder, use calibrated values - Now from CameraParams
    CALIBRATION_FILE = "data/camera_calibration.yaml"
    
    # This is a CRITICAL parameter and needs to be tuned based on MiDaS model and scene.
    # MiDaS produces relative depth. This factor is now passed to estimate_depth.
    # The appropriate value depends on the scene, model, and desired metric scale.
    # For example, if MiDaS output (after its internal normalization) is roughly in [0,1]
    # and represents depths up to, say, 10 meters, then a scale factor of 10 might be used.
    # This needs empirical tuning.
    DEPTH_OUTPUT_SCALE_FACTOR = 10.0 # Example: Assumes MiDaS output needs scaling by this for meters.
    DEFAULT_SAVE_PATH = "data/generated_map_tsdf.ply" # Save path for TSDF extracted cloud
    DEFAULT_LOAD_PATH = "data/generated_map_tsdf.ply" # Load path for a PLY file

    # --- TSDF Parameters for PointCloudMapper ---
    TSDF_VOXEL_LENGTH = 0.02  # meters, size of a TSDF voxel
    TSDF_SDF_TRUNC = 0.04     # meters, truncation distance for SDF
    # VOXEL_SIZE_POINT_CLOUD = 0.03 # meters - This is now deprecated in favor of TSDF params

    try:
        # --- Initialization ---
        print("Initializing camera...")
        camera = MonocularCamera(0) # Or specify a video file path

        print("Initializing Camera Parameters...")
        # cam_params = CameraParams(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, fx=FX, fy=FY)
        cam_params = CameraParams(calibration_file_path=CALIBRATION_FILE,
                                  default_image_width=IMAGE_WIDTH,
                                  default_image_height=IMAGE_HEIGHT)
        # Update IMAGE_WIDTH and IMAGE_HEIGHT from actual loaded params
        IMAGE_WIDTH, IMAGE_HEIGHT = cam_params.get_image_dimensions()
        print(f"Camera parameters initialized using '{CALIBRATION_FILE}'. Loaded K=\n{cam_params.get_K()}")
        print(f"Image dimensions from CameraParams: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")


        print("Initializing MiDaS Depth Estimator...")
        # The 'output_scale_factor' will be added to MiDaSDepthEstimator later
        # For now, we'll apply it externally.
        depth_estimator = MiDaSDepthEstimator()

        print("Initializing Visual Odometry...")
        vo = VisualOdometry(camera_params=cam_params)

        print("Initializing PointCloud Mapper with TSDF...")
        pointcloud_mapper = PointCloudMapper(camera_params=cam_params, 
                                             voxel_length_tsdf=TSDF_VOXEL_LENGTH, 
                                             sdf_trunc_tsdf=TSDF_SDF_TRUNC)
        # Initialize the temporary attribute for loaded PCDs in the mapper instance for clarity with example logic
        pointcloud_mapper._loaded_external_pcd = None


        print("Initializing Open3D Visualizer...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Global Point Cloud", width=800, height=600)
        vis.get_render_option().point_size = 2
        vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1]) # Dark background
        
        # Coordinate frame for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame) # Add once, it doesn't change

        # Load map if specified
        if args.load_map:
            print(f"Attempting to load map from command line argument: {args.load_map}")
            # The load_map in TSDF version now stores to _loaded_external_pcd
            if pointcloud_mapper.load_map(args.load_map) and pointcloud_mapper._loaded_external_pcd:
                pcd_vis_geom = pointcloud_mapper._loaded_external_pcd # Use the externally loaded PCD
                if pcd_vis_geom.has_points():
                    vis.add_geometry(pcd_vis_geom, reset_bounding_box=True)
                    print(f"Map from {args.load_map} loaded and added to visualizer.")
                    print("Note: Live TSDF processing will start with a fresh volume (if camera runs).")
                else:
                    print(f"Loaded map from {args.load_map} is empty.")
                    pcd_vis_geom = o3d.geometry.PointCloud()
                    vis.add_geometry(pcd_vis_geom, reset_bounding_box=True)
            else:
                print(f"Failed to load map from {args.load_map}. Starting with an empty map for visualization.")
                pcd_vis_geom = o3d.geometry.PointCloud()
                vis.add_geometry(pcd_vis_geom, reset_bounding_box=True)
        else:
            pcd_vis_geom = o3d.geometry.PointCloud() # Start with an empty cloud for TSDF to populate
            vis.add_geometry(pcd_vis_geom, reset_bounding_box=True)
            print("No initial map loaded. Live TSDF reconstruction will populate the map.")


        last_time = time.time()
        print("Press 'q' in the OpenCV window to quit.")
        print(f"Press 'k' to save the current TSDF-extracted map to {DEFAULT_SAVE_PATH}.")
        print(f"Press 'l' to load a PLY map from {DEFAULT_LOAD_PATH} (for visualization, TSDF will reset).")
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

            # --- Visual Odometry ---
            # R_w2c, t_w2c is the pose of the camera in the world (transforms world points to camera coords)
            R_w2c, t_w2c, vo_kps, vo_matches = vo.process_frame(frame_bgr)

            # --- Depth Estimation ---
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Assuming estimate_depth() will be modified to return scaled metric depth.
            # For now, simulate this:
            # The MiDaSDepthEstimator's estimate_depth method now takes an output_scale_factor.
            depth_map_m = depth_estimator.estimate_depth(frame_rgb, output_scale_factor=DEPTH_OUTPUT_SCALE_FACTOR)
            
            # Ensure depth is positive and clip very large values if necessary.
            # The depth_trunc in create_from_rgbd_image will also handle max depth.
            depth_map_m = np.clip(depth_map_m, 0.01, 30.0) # Clip depth (e.g. 1cm to 30m)


            # --- Point Cloud Generation and Mapping ---
            # R_w2c, t_w2c is from VO (world to camera)
            # This call integrates the frame into pointcloud_mapper.tsdf_volume
            pointcloud_mapper.integrate_frame_tsdf(frame_bgr, depth_map_m, R_w2c, t_w2c)
            
            # --- Visualization Update ---
            # Extract point cloud from TSDF for visualization
            current_pcd_from_tsdf = pointcloud_mapper.get_global_point_cloud()
            
            if current_pcd_from_tsdf.has_points():
                pcd_vis_geom.points = current_pcd_from_tsdf.points
                pcd_vis_geom.colors = current_pcd_from_tsdf.colors
                vis.update_geometry(pcd_vis_geom)
                # Reset bounding box only if it's the first substantial cloud, 
                # or if explicitly requested, to avoid view jumping.
                # if not vis.get_view_control().get_bounding_box().has_points() : # Heuristic
                #    vis.reset_view_point(True)
            else:
                pcd_vis_geom.clear() # Clear if TSDF is empty
                vis.update_geometry(pcd_vis_geom)

            vis.poll_events()
            vis.update_renderer()
            
            # Display camera feed and depth (optional, can be slow)
            # frame_with_kps = cv2.drawKeypoints(frame_bgr, vo_kps, None, color=(0,255,0))
            # cv2.putText(frame_with_kps, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            # cv2.imshow('Camera Feed', frame_with_kps)

            # depth_display = cv2.normalize(depth_map_m, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)
            # cv2.imshow('Depth Map Display', depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('k'):
                print(f"Saving map to {DEFAULT_SAVE_PATH}...")
                pointcloud_mapper.save_map(DEFAULT_SAVE_PATH)
            elif key == ord('l'):
                print(f"Attempting to load PLY map from {DEFAULT_LOAD_PATH} for visualization...")
                # Reset TSDF volume when loading an external map for pure visualization,
                # so subsequent live processing starts fresh.
                print("Resetting internal TSDF volume.")
                pointcloud_mapper.tsdf_volume.reset() 
                # Clear the current visualizer geometry
                if pcd_vis_geom:
                    pcd_vis_geom.clear()
                    vis.update_geometry(pcd_vis_geom) 

                if pointcloud_mapper.load_map(DEFAULT_LOAD_PATH) and pointcloud_mapper._loaded_external_pcd:
                    # Replace pcd_vis_geom's data with the loaded one
                    pcd_vis_geom.points = pointcloud_mapper._loaded_external_pcd.points
                    pcd_vis_geom.colors = pointcloud_mapper._loaded_external_pcd.colors
                    if pcd_vis_geom.has_points():
                        vis.update_geometry(pcd_vis_geom)
                        vis.reset_view_point(True) # Reset view to the loaded map
                        print("Loaded PLY map displayed. Live TSDF will start fresh if processing continues.")
                    else:
                        print("Loaded PLY map is empty.")
                        vis.update_geometry(pcd_vis_geom) # Show empty state
                else:
                    print(f"Failed to load PLY map from {DEFAULT_LOAD_PATH}.")
            elif key == ord('m'): # Extract and view mesh
                print("Extracting mesh from TSDF volume...")
                mesh = pointcloud_mapper.get_mesh()
                if mesh.has_vertices() and mesh.has_triangles():
                    print(f"Mesh extracted with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
                    # Display in a new window to avoid replacing the live point cloud view
                    o3d.visualization.draw_geometries([mesh], window_name="Extracted Mesh")
                else:
                    print("Could not extract a valid mesh from the TSDF volume (it might be too sparse).")


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
