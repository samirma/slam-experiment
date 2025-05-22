import unittest
import numpy as np
import yaml
import os
import sys
from pathlib import Path

# Add the src directory to the Python path to allow importing CameraParams
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.camera_params import CameraParams

class TestCameraParams(unittest.TestCase):
    """
    Unit tests for the CameraParams class from src.utils.camera_params.
    """
    TEST_DATA_DIR = "data_test_camera_params" # Temporary directory for test files
    VALID_CALIB_FILE = os.path.join(TEST_DATA_DIR, "valid_calibration.yaml")
    INVALID_CALIB_FILE_MISSING_KEYS = os.path.join(TEST_DATA_DIR, "invalid_calibration_missing.yaml")
    INVALID_CALIB_FILE_BAD_FORMAT = os.path.join(TEST_DATA_DIR, "invalid_calibration_bad_format.yaml")

    @classmethod
    def setUpClass(cls):
        """
        Set up a temporary directory and create dummy calibration files for testing.
        This method is called once before any tests in the class are run.
        """
        os.makedirs(cls.TEST_DATA_DIR, exist_ok=True)

        # Create a valid calibration file
        valid_data = {
            'camera_matrix': [[500.0, 0.0, 320.0], [0.0, 501.0, 240.0], [0.0, 0.0, 1.0]],
            'dist_coeffs': [[0.1, -0.05, 0.001, 0.001, 0.02]], # OpenCV format (1xN or Nx1)
            'image_width': 640,
            'image_height': 480
        }
        with open(cls.VALID_CALIB_FILE, 'w') as f:
            yaml.dump(valid_data, f)

        # Create an invalid calibration file (missing keys)
        invalid_data_missing = {
            'camera_matrix': [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
            # dist_coeffs is missing
        }
        with open(cls.INVALID_CALIB_FILE_MISSING_KEYS, 'w') as f:
            yaml.dump(invalid_data_missing, f)

        # Create an invalid calibration file (bad format)
        with open(cls.INVALID_CALIB_FILE_BAD_FORMAT, 'w') as f:
            f.write("camera_matrix: [1, 2, 3]\ndist_coeffs: not_a_list_at_all") # Malformed YAML / data

    @classmethod
    def tearDownClass(cls):
        """
        Clean up by removing the temporary directory and files after all tests.
        This method is called once after all tests in the class have run.
        """
        if os.path.exists(cls.VALID_CALIB_FILE):
            os.remove(cls.VALID_CALIB_FILE)
        if os.path.exists(cls.INVALID_CALIB_FILE_MISSING_KEYS):
            os.remove(cls.INVALID_CALIB_FILE_MISSING_KEYS)
        if os.path.exists(cls.INVALID_CALIB_FILE_BAD_FORMAT):
            os.remove(cls.INVALID_CALIB_FILE_BAD_FORMAT)
        if os.path.exists(cls.TEST_DATA_DIR):
            os.rmdir(cls.TEST_DATA_DIR)

    def test_load_valid_calibration_file(self):
        """Test loading parameters from a correctly formatted YAML file."""
        params = CameraParams(calibration_file_path=self.VALID_CALIB_FILE)
        self.assertIsNotNone(params.get_K(), "Camera matrix K should be loaded.")
        self.assertIsNotNone(params.get_dist_coeffs(), "Distortion coeffs should be loaded.")
        self.assertEqual(params.get_K()[0,0], 500.0)
        self.assertEqual(params.get_K()[1,1], 501.0)
        self.assertEqual(params.get_image_dimensions(), (640, 480))
        self.assertEqual(params.get_dist_coeffs().shape[0], 1) # Check if it's a row vector
        self.assertEqual(params.get_dist_coeffs().shape[1], 5) # Check if it has 5 elements

    def test_no_calibration_file_uses_placeholders(self):
        """Test that placeholder values are used if no calibration file is provided."""
        params = CameraParams(default_image_width=640, default_image_height=480)
        self.assertIsNotNone(params.get_K(), "K should have placeholder values.")
        self.assertIsNotNone(params.get_dist_coeffs(), "dist_coeffs should have placeholder values.")
        # Check if default fx, fy from _initialize_placeholders are used (e.g. 550.0)
        # These might change if the placeholder defaults change, so test against known placeholder values
        self.assertEqual(params.get_K()[0,0], 550.0) # Default placeholder fx
        self.assertEqual(params.get_K()[0,2], 320.0) # cx = default_image_width / 2
        self.assertEqual(params.get_dist_coeffs().shape, (5,1)) # Default placeholder shape
        self.assertTrue(np.all(params.get_dist_coeffs() == 0)) # Default placeholder is zeros

    def test_non_existent_calibration_file_uses_placeholders(self):
        """Test that placeholders are used if the specified calibration file does not exist."""
        params = CameraParams(calibration_file_path="data/non_existent_file.yaml",
                              default_image_width=320, default_image_height=240)
        self.assertIsNotNone(params.get_K(), "K should have placeholder values.")
        self.assertIsNotNone(params.get_dist_coeffs(), "dist_coeffs should have placeholder values.")
        self.assertEqual(params.get_K()[0,0], 550.0) # Default placeholder fx
        self.assertEqual(params.get_K()[0,2], 160.0) # cx = default_image_width / 2
        self.assertEqual(params.get_image_dimensions(), (320,240))


    def test_invalid_calibration_file_missing_keys(self):
        """Test that placeholders are used if the calibration file is missing required keys."""
        params = CameraParams(calibration_file_path=self.INVALID_CALIB_FILE_MISSING_KEYS)
        self.assertIsNotNone(params.get_K(), "K should have placeholder values after failed load.")
        # Check a known placeholder value (e.g., fx=550.0)
        self.assertEqual(params.get_K()[0,0], 550.0) 

    def test_invalid_calibration_file_bad_format(self):
        """Test that placeholders are used if the calibration file is malformed."""
        params = CameraParams(calibration_file_path=self.INVALID_CALIB_FILE_BAD_FORMAT)
        self.assertIsNotNone(params.get_K(), "K should have placeholder values after failed load due to bad format.")
        self.assertEqual(params.get_K()[0,0], 550.0)

    def test_get_K_returns_correct_type_and_shape(self):
        """Test that get_K() returns a NumPy array with shape (3,3)."""
        params = CameraParams(calibration_file_path=self.VALID_CALIB_FILE)
        K = params.get_K()
        self.assertIsInstance(K, np.ndarray)
        self.assertEqual(K.shape, (3,3))

    def test_get_dist_coeffs_returns_correct_type_and_shape(self):
        """Test that get_dist_coeffs() returns a NumPy array."""
        params = CameraParams(calibration_file_path=self.VALID_CALIB_FILE)
        dist = params.get_dist_coeffs()
        self.assertIsInstance(dist, np.ndarray)
        # Shape can vary (e.g., (1,5), (5,1), (1,4) etc.)
        # For the valid_data, it's [[...]] which becomes (1,5)
        self.assertTrue(dist.ndim == 2 and (dist.shape[0] == 1 or dist.shape[1] == 1))
        self.assertGreaterEqual(dist.size, 4) # Should have at least k1,k2,p1,p2

    def test_image_dimensions_loaded_correctly(self):
        """Test that image dimensions are loaded from file or defaults used."""
        params_loaded = CameraParams(calibration_file_path=self.VALID_CALIB_FILE)
        width, height = params_loaded.get_image_dimensions()
        self.assertEqual(width, 640)
        self.assertEqual(height, 480)

        params_default = CameraParams(default_image_width=1280, default_image_height=720)
        width_def, height_def = params_default.get_image_dimensions()
        self.assertEqual(width_def, 1280)
        self.assertEqual(height_def, 720)

if __name__ == '__main__':
    unittest.main()
