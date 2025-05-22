import numpy as np
import open3d as o3d
import cv2 # For BGR to RGB conversion
import os

# Need to ensure src is in path for this import to work directly if run as script
# In a project context, this should be fine with proper PYTHONPATH or execution from root
try:
    from src.utils import CameraParams
except ImportError:
    print("Attempting fallback import for CameraParams. Ensure PYTHONPATH includes 'src' or run from project root.")
    # This is a simplified fallback. In a real project, ensure your environment is set up.
    # For development, you might add 'src' to sys.path here if needed.
    from utils.camera_params import CameraParams # Placeholder if previous fails

class PointCloudMapper:
    """
    Manages 3D reconstruction using a Truncated Signed Distance Function (TSDF) volume.
    It integrates RGBD frames (color and depth) along with camera poses into a global
    TSDF volume, from which a point cloud or mesh can be extracted.
    """
    def __init__(self, camera_params: CameraParams, 
                 voxel_length: float = 0.02, 
                 sdf_trunc: float = 0.04):
        """
        Initializes the PointCloudMapper with TSDF integration.

        Args:
            camera_params (CameraParams): An object containing the camera's intrinsic parameters
                                          (matrix K, distortion coefficients, image dimensions).
            voxel_length (float): The length of a single voxel in the TSDF volume, in meters.
                                  Smaller values result in higher resolution but more memory usage.
                                  Defaults to 0.02.
            sdf_trunc (float): The truncation distance for the Signed Distance Function (SDF), in meters.
                               This defines the thickness of the surface region being modeled.
                               Typically set to a few times the voxel_length. Defaults to 0.04.
        """
        self.camera_params: CameraParams = camera_params
        
        # --- TSDF Volume Parameters ---
        self.voxel_length: float = voxel_length  # Voxel size in meters
        self.sdf_trunc: float = sdf_trunc      # SDF truncation distance in meters
        
        # Initialize the ScalableTSDFVolume. This volume can dynamically expand as more data is integrated.
        # Color type is set to RGB8, meaning it will store 8-bit RGB color information.
        self.tsdf_volume: o3d.pipelines.integration.ScalableTSDFVolume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # Temporary storage for a point cloud loaded externally via load_map()
        self._loaded_external_pcd: o3d.geometry.PointCloud | None = None

        # Get camera intrinsics once
        self.K: np.ndarray = self.camera_params.get_K()
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.image_width, self.image_height = self.camera_params.get_image_dimensions()

        self.o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.image_width,
            height=self.image_height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy
        )

    def _integrate_rgbd_into_tsdf(self, color_frame_bgr: np.ndarray, depth_map_m: np.ndarray,
                                  camera_pose_R_world_to_cam: np.ndarray, 
                                  camera_pose_t_world_to_cam: np.ndarray) -> None:
        """
        Internal method to integrate a single RGBD frame into the TSDF volume.

        Args:
            color_frame_bgr (numpy.ndarray): The BGR color image (H, W, C).
            depth_map_m (numpy.ndarray): The depth map in meters (H, W).
            camera_pose_R_world_to_cam (numpy.ndarray): Rotation matrix (3x3) representing
                                                       the orientation of the world in the current camera frame (R_w2c).
            camera_pose_t_world_to_cam (numpy.ndarray): Translation vector (3x1) representing
                                                       the position of the world origin in the current camera frame (t_w_in_c).
        
        Raises:
            ValueError: If color and depth dimensions do not match, or if they don't match
                        the dimensions set in `camera_params`.
        """
        if color_frame_bgr.shape[:2] != depth_map_m.shape[:2]:
            raise ValueError(f"Color frame ({color_frame_bgr.shape[:2]}) and depth map ({depth_map_m.shape[:2]}) dimensions must match.")
        if color_frame_bgr.shape[0] != self.image_height or color_frame_bgr.shape[1] != self.image_width:
             print(f"Warning: Input frame dimensions ({color_frame_bgr.shape[1]}x{color_frame_bgr.shape[0]}) " \
                   f"differ from CameraParams dimensions ({self.image_width}x{self.image_height}). " \
                   "This may lead to incorrect projection if intrinsics are not scaled accordingly. " \
                   "Ensure input frames match the dimensions used for o3d_intrinsics or resize them.")

        # Convert BGR color frame to RGB for Open3D
        color_frame_rgb = cv2.cvtColor(color_frame_bgr, cv2.COLOR_BGR2RGB)

        # Create Open3D Image objects from NumPy arrays
        depth_o3d = o3d.geometry.Image(depth_map_m.astype(np.float32))
        color_o3d = o3d.geometry.Image(color_frame_rgb.astype(np.uint8))

        # Create an RGBDImage object.
        # depth_scale=1.0 because depth_map_m is already in meters.
        # depth_trunc truncates depth values beyond this distance (in meters) for TSDF integration.
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,  # Depth is already in meters
            depth_trunc=self.sdf_trunc * 5.0, # Heuristic: truncate depth further than SDF truncation
            convert_rgb_to_intensity=False # Keep color information
        )

        # Calculate the camera-to-world extrinsic matrix for TSDF integration.
        # The TSDF volume `integrate` function expects the extrinsic matrix that transforms
        # points from the camera coordinate system to the world coordinate system (T_camera_to_world).
        # Visual Odometry (VO) typically provides the world-to-camera transformation (T_world_to_camera).
        # T_camera_to_world = (T_world_to_camera)^-1
        # If T_world_to_camera = [R_w2c | t_w_in_c], then T_camera_to_world = [R_w2c.T | -R_w2c.T @ t_w_in_c]
        
        R_cam_to_world = camera_pose_R_world_to_cam.T
        t_cam_to_world = -np.dot(camera_pose_R_world_to_cam.T, camera_pose_t_world_to_cam)

        extrinsic_cam_to_world = np.eye(4, dtype=np.float64) # Must be float64 for integrate
        extrinsic_cam_to_world[0:3, 0:3] = R_cam_to_world
        extrinsic_cam_to_world[0:3, 3] = t_cam_to_world.ravel() # .ravel() ensures (3,) shape
        
        # Integrate the RGBD frame into the TSDF volume
        self.tsdf_volume.integrate(
            rgbd_image,
            self.o3d_intrinsics, # PinholeCameraIntrinsic
            extrinsic_cam_to_world # Extrinsic matrix (camera to world)
        )
        
        # After integration, _loaded_external_pcd should be cleared if it exists,
        # as the TSDF volume is now the primary source of the map.
        if self._loaded_external_pcd is not None:
            print("Clearing previously loaded external PCD as new data has been integrated into TSDF.")
            self._loaded_external_pcd = None


    def integrate_frame_tsdf(self, color_frame_bgr: np.ndarray, depth_map_m: np.ndarray,
                                camera_pose_R_world_to_cam: np.ndarray, 
                                camera_pose_t_world_to_cam: np.ndarray) -> None:
        """
        Integrates a new frame (color and depth) into the TSDF volume, using its estimated camera pose.

        Args:
            color_frame_bgr (numpy.ndarray): The input BGR color image (H, W, C).
            depth_map_m (numpy.ndarray): The corresponding depth map, with depth values in meters (H, W).
            camera_pose_R_world_to_cam (numpy.ndarray): The 3x3 rotation matrix from Visual Odometry,
                                                       representing the orientation of the world
                                                       in the current camera's coordinate system (R_w2c).
            camera_pose_t_world_to_cam (numpy.ndarray): The 3x1 translation vector from Visual Odometry,
                                                       representing the position of the world origin
                                                       in the current camera's coordinate system (t_w_in_c).
        """
        self._integrate_rgbd_into_tsdf(color_frame_bgr, depth_map_m, 
                                       camera_pose_R_world_to_cam, camera_pose_t_world_to_cam)


    def get_global_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        Extracts and returns the current global point cloud from the TSDF volume.
        If an external point cloud was loaded via `load_map` and no new frames have been
        integrated into TSDF since, the loaded point cloud is returned.

        Returns:
            open3d.geometry.PointCloud: The extracted point cloud. Can be empty if
                                        TSDF volume is empty or no external cloud loaded.
        """
        if self._loaded_external_pcd is not None:
            # print("Returning externally loaded point cloud.")
            return self._loaded_external_pcd
        
        # print("Extracting point cloud from TSDF volume...")
        pcd = self.tsdf_volume.extract_point_cloud()
        # print(f"Extracted point cloud with {len(pcd.points)} points from TSDF.")
        return pcd
    
    def get_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        Extracts a triangle mesh representation from the TSDF volume.
        The mesh is generated using the Marching Cubes algorithm on the TSDF data.
        Vertex normals are computed for the extracted mesh.

        Returns:
            open3d.geometry.TriangleMesh: The extracted mesh. Can be empty if TSDF volume is empty.
        """
        mesh = self.tsdf_volume.extract_triangle_mesh()
        if mesh.has_vertices(): # Only compute normals if mesh is not empty
            mesh.compute_vertex_normals()
        # print(f"Extracted mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles from TSDF.")
        return mesh

    def save_map(self, file_path: str) -> None:
        """
        Saves the current global point cloud (extracted from the TSDF volume) to a .ply file.

        Args:
            file_path (str): The path (including filename, e.g., "map.ply") where the
                             point cloud will be saved.
        """
        pcd_to_save = self.tsdf_volume.extract_point_cloud()
        if not pcd_to_save.has_points():
            print(f"Warning: TSDF volume is empty or point cloud extraction yielded no points. "
                  f"Saving an empty file to {file_path}.")
        
        try:
            # Ensure the parent directory for the file_path exists
            dir_name = os.path.dirname(file_path)
            if dir_name: # Check if dirname is not empty (i.e., not saving in current dir)
                os.makedirs(dir_name, exist_ok=True)

            success = o3d.io.write_point_cloud(file_path, pcd_to_save)
            if success:
                print(f"TSDF-extracted point cloud saved successfully to {file_path}")
            else:
                print(f"Failed to save TSDF-extracted point cloud to {file_path}.")
        except Exception as e:
            print(f"An error occurred while saving the TSDF-extracted map: {e}")

    def load_map(self, file_path: str) -> bool:
        """
        Loads a point cloud from a file. 
        NOTE: This loaded point cloud is NOT integrated back into the TSDF volume.
        The TSDF volume will be reset if live processing continues.
        This method primarily serves to load a PLY for visualization if needed.

        Args:
            file_path (str): The path to the point cloud file to load.

        Returns:
            bool: True if loading was successful and map has points, False otherwise.
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return False
        
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if pcd is not None and pcd.has_points():
                # self.global_point_cloud = pcd # This attribute is no longer the primary store.
                # Instead of storing it, perhaps this method should just return it,
                # and the caller decides what to do (e.g. visualize it).
                # For now, let's print a message and return true. The caller can get it via get_global_point_cloud
                # if we decide to replace the TSDF (which we are not doing here).
                print(f"Point cloud loaded successfully from {file_path}. Points: {len(pcd.points)}")
                print("Note: This loaded map does not re-initialize the internal TSDF volume for further integration.")
                print("If live processing continues, TSDF will build upon its current state or a new empty volume.")
                # To allow visualization of this loaded cloud via get_global_point_cloud temporarily,
                # one could replace the TSDF volume here, or store this pcd in a temporary variable.
                # For simplicity, loading a map means subsequent get_global_point_cloud will return this loaded cloud
                # until new frames are integrated, which will then use a new TSDF.
                # This is tricky. Let's make load_map reset the TSDF volume and integrate this pcd as a "pseudo-frame".
                # This is complex.
                # Simpler: load_map just loads a point cloud, returns it. The main script will handle visualization.
                # The PointCloudMapper's internal TSDF is NOT changed by this load.
                #
                # Re-decision: load_map is problematic with TSDF.
                # For this step, let's make load_map load the points and store them in a temporary
                # variable that get_global_point_cloud can return if no TSDF integration has happened since load.
                # This is still messy.
                #
                # Simplest for now: load_map loads a point cloud and returns it.
                # The PointCloudMapper itself will NOT store this loaded cloud as its "global_point_cloud".
                # The caller (main script) will decide what to do with it.
                # This means the 'l' key in run_pointcloud_generation.py will need to handle this.
                # For now, let's keep the old behavior: it loads a point cloud, but it's disconnected from TSDF.
                # The `global_point_cloud` attribute can be used as a temporary holder for a loaded cloud.
                # self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(...) # Reset TSDF
                # One option: make self.global_point_cloud store the externally loaded map.
                # get_global_point_cloud would then need logic: if self.global_point_cloud exists (from load) and no new TSDF data, return it.
                # Else, extract from TSDF.
                # This attribute is used by get_global_point_cloud to return this loaded cloud
                # if no new frames have been integrated into TSDF since loading.
                self._loaded_external_pcd = pcd 
                return True
            else: # pcd is None or has no points
                print(f"Failed to load a valid point cloud from {file_path}. File might be empty or corrupted.")
                self._loaded_external_pcd = None
                return False
        except Exception as e:
            print(f"An error occurred while loading the map from {file_path}: {e}")
            self._loaded_external_pcd = None
            return False

if __name__ == '__main__':
    # This block serves as an example of how to use the PointCloudMapper class.
    # It demonstrates initialization, frame integration, point cloud/mesh extraction, and saving/loading.
    # Note: This example requires a GUI environment if visualization commands (o3d.visualization.draw_geometries) are uncommented.
    
    print("PointCloudMapper Example Usage with TSDF")

    # 1. Setup Dummy Camera Parameters
    IMG_WIDTH, IMG_HEIGHT = 640, 480
    FX, FY = 550.0, 550.0 
    CX, CY = IMG_WIDTH / 2, IMG_HEIGHT / 2
    
    class DummyCameraParams: 
        def get_K(self) -> np.ndarray: 
            return np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)
        def get_dist_coeffs(self) -> np.ndarray: 
            return np.zeros((5,1), dtype=np.float32)
        def get_image_dimensions(self) -> tuple[int, int]: 
            return IMG_WIDTH, IMG_HEIGHT

    cam_params_dummy = DummyCameraParams()
    
    # Initialize PointCloudMapper with TSDF parameters
    mapper = PointCloudMapper(camera_params=cam_params_dummy, voxel_length=0.02, sdf_trunc=0.04)
    # mapper._loaded_external_pcd is initialized to None by __init__

    # 2. Create dummy RGBD data for a frame
    # Dummy BGR color image (random noise)
    dummy_color_bgr = np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    # Dummy depth map (e.g., a ramp from 0.5m to 2.5m, in meters)
    dummy_depth_m = np.fromfunction(lambda r, c: (r + c) / (IMG_HEIGHT + IMG_WIDTH) * 2.0 + 0.5, 
                                    (IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    dummy_depth_m = np.clip(dummy_depth_m, 0.1, 5.0) # Clip depth values

    # 3. Dummy camera pose (world-to-camera) for the first frame (identity pose)
    R_w2c_frame1 = np.eye(3, dtype=np.float32)
    t_w_in_c_frame1 = np.zeros((3, 1), dtype=np.float32)

    print("Integrating first frame into TSDF...")
    mapper.integrate_frame_tsdf(dummy_color_bgr, dummy_depth_m, R_w2c_frame1, t_w_in_c_frame1)
    
    # Extract and check point cloud from TSDF
    pcd_after_frame1 = mapper.get_global_point_cloud()
    print(f"Point cloud after first frame has {len(pcd_after_frame1.points)} points.")

    # 4. Simulate a second frame with a new camera pose
    # Example: Camera moved slightly along its X-axis (world's X if R_w2c is identity)
    # and rotated slightly. This R_w2c_frame2, t_w_in_c_frame2 is the new world-to-camera pose.
    # For simplicity, let's assume a small translation and rotation from the first pose.
    # R_prev_to_curr_example = cv2.Rodrigues(np.array([0, 0.1, 0]))[0] # Small rotation
    # t_curr_in_prev_example = np.array([[0.1],[0],[0]], dtype=np.float32) # Small translation
    # R_w2c_frame2 = R_prev_to_curr_example @ R_w2c_frame1
    # t_w_in_c_frame2 = R_prev_to_curr_example @ t_w_in_c_frame1 + t_curr_in_prev_example
    # For this simple example, let's just define a slightly different world-to-camera pose:
    R_w2c_frame2 = cv2.Rodrigues(np.array([0, np.pi/20, 0]))[0].astype(np.float32) # Rotated by ~9 deg around Y
    t_w_in_c_frame2 = np.array([[0.1], [0], [0.05]], dtype=np.float32)
    
    dummy_color_bgr_2 = np.roll(dummy_color_bgr, shift=(50,50), axis=(0,1)) # Slightly different color image
    dummy_depth_m_2 = dummy_depth_m * 0.9 # Objects appear slightly closer

    print("\nIntegrating second frame into TSDF...")
    mapper.integrate_frame_tsdf(dummy_color_bgr_2, dummy_depth_m_2, R_w2c_frame2, t_w_in_c_frame2)
    pcd_after_frame2 = mapper.get_global_point_cloud()
    print(f"Point cloud after second frame has {len(pcd_after_frame2.points)} points.")

    # 5. Test saving the TSDF-extracted point cloud
    SAVE_TEST_PATH_TSDF = "data/test_map_tsdf.ply" 
    print(f"\nAttempting to save TSDF-extracted map to {SAVE_TEST_PATH_TSDF}...")
    mapper.save_map(SAVE_TEST_PATH_TSDF)

    # 6. Test loading an external PLY map (for visualization purposes)
    # This uses the map just saved.
    print(f"\nAttempting to load map from {SAVE_TEST_PATH_TSDF} (for visualization, does not affect internal TSDF)...")
    load_success = mapper.load_map(SAVE_TEST_PATH_TSDF) 
    if load_success and mapper._loaded_external_pcd: # Access the temporary attribute
        loaded_pcd_for_viz = mapper._loaded_external_pcd
        print(f"Map loaded for visualization. Points: {len(loaded_pcd_for_viz.points)}")
        # In a real application, the main script (`run_pointcloud_generation.py`) would handle
        # the visualization of this `loaded_pcd_for_viz`.
        # e.g., o3d.visualization.draw_geometries([loaded_pcd_for_viz], window_name="Loaded Test Map (External)")
    else:
        print(f"Failed to load map for visualization from {SAVE_TEST_PATH_TSDF}.")

    # 7. Test mesh extraction
    print("\nExtracting mesh from TSDF volume...")
    mesh_from_tsdf = mapper.get_mesh()
    if mesh_from_tsdf.has_vertices():
        print(f"Mesh extracted with {len(mesh_from_tsdf.vertices)} vertices and {len(mesh_from_tsdf.triangles)} triangles.")
        # e.g., o3d.visualization.draw_geometries([mesh_from_tsdf], window_name="Extracted TSDF Mesh")
    else:
        print("No mesh could be extracted from TSDF (volume might be too sparse).")

    print("\nPointCloudMapper TSDF example usage finished.")
    print(f"Note: Test file '{SAVE_TEST_PATH_TSDF}' was created/overwritten.")
