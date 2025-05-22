import tensorflow as tf
# tensorflow_hub is not directly used for loading anymore, but good to list if other parts rely on TF Hub concepts
# import tensorflow_hub as hub 
import numpy as np
import cv2
import os
import requests
import tarfile
import shutil # For cleanup if needed

class MiDaSDepthEstimator:
    """
    A class to estimate depth from an RGB image using a MiDaS model.
    The model is downloaded from a specified URL (if not already cached locally)
    and loaded using TensorFlow's SavedModel format.
    It handles model loading, image preprocessing, inference, and postprocessing.
    """
    def __init__(self, 
                 model_url: str = "https://tfhub.dev/intel/midas/v2_1/1?tf-hub-format=compressed",
                 model_name: str = "midas_v2_1_tfhub_large", # Used for default local path subdirectory
                 models_base_dir: str = "models"): # Base directory for all models
        """
        Initializes the MiDaS depth estimator.
        Ensures the model is available locally (downloads if not) and then loads it.

        Args:
            model_url (str, optional): URL to download the MiDaS model (tar.gz format).
                                       Defaults to the MiDaS v2.1 large model from TF Hub.
            model_name (str, optional): A name for the model, used to create a subdirectory
                                        within `models_base_dir` for storing this model's files.
                                        Defaults to "midas_v2_1_tfhub_large".
            models_base_dir (str, optional): The base directory where model subdirectories will be stored.
                                             Defaults to "models" in the current working directory.
        
        Raises:
            RuntimeError: If the model cannot be downloaded, extracted, or loaded.
        """
        self.model_url: str = model_url
        self.model_name: str = model_name
        
        # self.model_storage_dir is where this specific model's archive is downloaded and extracted.
        # e.g., current_working_dir/models/midas_v2_1_tfhub_large/
        # The actual SavedModel might be directly in it or in a subdirectory (e.g., .../1/)
        self.model_storage_dir: str = os.path.join(os.getcwd(), models_base_dir, self.model_name)

        self.model: tf.keras.Model | None = None 
        self.expected_input_height: int = 384 # Default for MiDaS v2.1
        self.expected_input_width: int = 384  # Default for MiDaS v2.1

        try:
            # _ensure_model_is_local will download if needed and return the path to the
            # directory containing the 'saved_model.pb' file.
            actual_model_load_path = self._ensure_model_is_local(self.model_url, self.model_storage_dir)
            
            print(f"Loading MiDaS model from resolved local path: {actual_model_load_path}...")
            self.model = tf.saved_model.load(actual_model_load_path)
            print("MiDaS model loaded successfully.")

            # MiDaS models are generally flexible with input size, but often perform best or are
            # trained with specific aspect ratios or sizes. 384x384 is common for v2.1.
            # We'll stick to this default for preprocessing.
            print(f"Using expected input size (H,W): ({self.expected_input_height}, {self.expected_input_width}) for preprocessing.")

        except Exception as e:
            print(f"Error initializing MiDaS model: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize MiDaS model: {e}")

    def _find_saved_model_path(self, base_dir: str) -> str | None:
        """
        Searches for 'saved_model.pb' in base_dir or one level of common subdirectories
        (like a version number, e.g., '1', or sometimes 'variables' if it's a Keras model saved differently).
        Returns the path to the directory containing 'saved_model.pb', or None if not found.
        """
        # Check base_dir itself
        if os.path.exists(os.path.join(base_dir, "saved_model.pb")):
            return base_dir
        
        # Check common subdirectories (e.g., version string like '1', '001')
        if os.path.isdir(base_dir): # Ensure base_dir exists and is a directory
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                # Check if item_path is a directory and contains saved_model.pb
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "saved_model.pb")):
                    return item_path # This is the actual SavedModel directory
        return None

    def _ensure_model_is_local(self, model_url: str, model_storage_dir: str) -> str:
        """
        Ensures the TensorFlow SavedModel is available. If not found in `model_storage_dir`
        (or its versioned subdirectories), it downloads and extracts the model from `model_url`.

        Args:
            model_url (str): The URL to download the .tar.gz model archive.
            model_storage_dir (str): The directory where the model files (including any version subdirs)
                                     will be stored (e.g., "models/midas_v2_1_tfhub_files").

        Returns:
            str: The path to the actual SavedModel directory (e.g., "models/midas_v2_1_tfhub_files/1").

        Raises:
            RuntimeError: If download, extraction, or finding the model fails.
        """
        os.makedirs(model_storage_dir, exist_ok=True)
        
        found_model_load_path = self._find_saved_model_path(model_storage_dir)
        if found_model_load_path:
            print(f"MiDaS model found locally at {found_model_load_path}")
            return found_model_load_path

        print(f"MiDaS model not found in {model_storage_dir} or its subdirectories. Downloading from {model_url}...")
        
        archive_filename = model_url.split("/")[-1].split("?")[0] 
        if not (archive_filename.endswith(".tar.gz") or archive_filename.endswith(".tgz")):
            archive_filename = "model.tar.gz" 
        
        temp_archive_path = os.path.join(model_storage_dir, archive_filename)

        try:
            print(f"Downloading to {temp_archive_path}...")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(temp_archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")

            print(f"Extracting archive {temp_archive_path} to {model_storage_dir}...")
            with tarfile.open(temp_archive_path, "r:gz") as tar:
                tar.extractall(path=model_storage_dir)
            print("Extraction complete.")

            # After extraction, search again to find the actual SavedModel directory path
            # This handles cases where the tarball contains a versioned subdirectory.
            final_model_load_path = self._find_saved_model_path(model_storage_dir)
            if not final_model_load_path:
                print(f"Contents of {model_storage_dir} after extraction: {os.listdir(model_storage_dir)}")
                raise RuntimeError(f"saved_model.pb not found in {model_storage_dir} or its subdirectories after extraction. Check tarball structure.")
            
            print(f"MiDaS model is now available locally at {final_model_load_path}")
            return final_model_load_path

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download model from {model_url}: {e}")
        except tarfile.TarError as e:
            raise RuntimeError(f"Failed to extract model archive {temp_archive_path}: {e}")
        except Exception as e:
            if os.path.exists(temp_archive_path) and os.path.isfile(temp_archive_path):
                try:
                    os.remove(temp_archive_path)
                except OSError as rm_err:
                    print(f"Warning: Could not remove temporary archive {temp_archive_path}: {rm_err}")
            raise RuntimeError(f"An error occurred during model setup: {e}")
        finally:
            if os.path.exists(temp_archive_path) and os.path.isfile(temp_archive_path):
                try:
                    os.remove(temp_archive_path)
                except OSError as rm_err:
                    print(f"Warning: Could not remove temporary archive {temp_archive_path} in finally: {rm_err}")

    def _preprocess_image(self, image_rgb: np.ndarray) -> tf.Tensor:
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        img_resized = cv2.resize(image_rgb, (self.expected_input_width, self.expected_input_height), 
                                 interpolation=cv2.INTER_CUBIC)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = tf.convert_to_tensor(img_normalized)[tf.newaxis, ...]
        return img_tensor

    def _postprocess_output(self, depth_tensor: tf.Tensor, 
                            original_height: int, original_width: int) -> np.ndarray:
        depth_map_model_dims = tf.squeeze(depth_tensor).numpy()
        depth_map_resized = cv2.resize(depth_map_model_dims, (original_width, original_height), 
                                       interpolation=cv2.INTER_CUBIC)
        return depth_map_resized

    def estimate_depth(self, image_rgb: np.ndarray, output_scale_factor: float = 1.0) -> np.ndarray:
        if image_rgb is None:
            raise ValueError("Input image cannot be None.")
        if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel RGB NumPy array (H, W, C).")

        original_height, original_width, _ = image_rgb.shape
        if self.model is None:
            raise RuntimeError("MiDaS model is not loaded. Cannot perform depth estimation.")

        preprocessed_image = self._preprocess_image(image_rgb)

        try:
            prediction = self.model(preprocessed_image) 
            
            if isinstance(prediction, dict):
                possible_keys = [
                    'default', 'output_1', 'prediction', 'depth_map', 
                    'midas_v21_prediction', 'midas_v21_small_prediction', 
                    'StatefulPartitionedCall:0', 
                ]
                depth_tensor = None
                for key_candidate in possible_keys:
                    if key_candidate in prediction:
                        depth_tensor = prediction[key_candidate]
                        break
                if depth_tensor is None:
                    if len(prediction) == 1:
                        depth_tensor = next(iter(prediction.values()))
                    else:
                        print(f"Warning: Model output dictionary keys: {list(prediction.keys())}")
                        raise RuntimeError("Could not find a known depth tensor key in model output dictionary.")
            elif isinstance(prediction, tf.Tensor):
                depth_tensor = prediction
            elif isinstance(prediction, list) and len(prediction) > 0 and isinstance(prediction[0], tf.Tensor):
                depth_tensor = prediction[0]
                if len(prediction) > 1:
                    print(f"Warning: Model output is a list of tensors, using the first one. Count: {len(prediction)}")
            else:
                raise RuntimeError(f"Model output type is not recognized: {type(prediction)}. Expected dict, Tensor, or list of Tensors.")

        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError(f"Model inference failed: {e}")

        depth_map = self._postprocess_output(depth_tensor, original_height, original_width)
        scaled_depth_map = depth_map * output_scale_factor
        return scaled_depth_map

if __name__ == '__main__':
    print("MiDaSDepthEstimator - Example Usage")
    
    # This is the base directory where 'midas_v2_1_tfhub_large' and 'midas_v2_1_tfhub_small'
    # subdirectories will be created by the MiDaSDepthEstimator instances.
    example_main_models_base_dir = "models" 
    if not os.path.exists(example_main_models_base_dir):
        try:
            os.makedirs(example_main_models_base_dir)
            print(f"Created base directory for models: {example_main_models_base_dir}")
        except OSError as e:
            print(f"Failed to create base directory {example_main_models_base_dir}: {e}")
            # Depending on desired behavior, might exit or raise
    
    try:
        dummy_image_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"Created a dummy RGB image of shape: {dummy_image_rgb.shape}")

        print("\nInitializing MiDaSDepthEstimator (default large model)...")
        estimator_large = MiDaSDepthEstimator(
            model_url="https://tfhub.dev/intel/midas/v2_1/1?tf-hub-format=compressed",
            model_name="midas_v2_1_tfhub_large", # Specific subdir for this model
            models_base_dir=example_main_models_base_dir
        )
        
        relative_depth_map_large = estimator_large.estimate_depth(dummy_image_rgb)
        print(f"Large model - Relative depth map stats: min={np.min(relative_depth_map_large):.2f}, max={np.max(relative_depth_map_large):.2f}")

        print("\nInitializing MiDaSDepthEstimator (small model)...")
        estimator_small = MiDaSDepthEstimator(
            model_url="https://tfhub.dev/intel/midas/v2_1_small/1?tf-hub-format=compressed",
            model_name="midas_v2_1_tfhub_small", # Specific subdir for this model
            models_base_dir=example_main_models_base_dir
        )
        relative_depth_map_small = estimator_small.estimate_depth(dummy_image_rgb)
        print(f"Small model - Relative depth map stats: min={np.min(relative_depth_map_small):.2f}, max={np.max(relative_depth_map_small):.2f}")

        # Optional: Display results if a GUI is available
        # depth_display_large = cv2.normalize(relative_depth_map_large, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # depth_colormap_large = cv2.applyColorMap(depth_display_large, cv2.COLORMAP_INFERNO)
        # cv2.imshow("Dummy RGB", cv2.cvtColor(dummy_image_rgb, cv2.COLOR_RGB2BGR))
        # cv2.imshow("Depth (Large Model)", depth_colormap_large)
        
        # depth_display_small = cv2.normalize(relative_depth_map_small, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # depth_colormap_small = cv2.applyColorMap(depth_display_small, cv2.COLORMAP_INFERNO)
        # cv2.imshow("Depth (Small Model)", depth_colormap_small)
        
        # print("Press any key to close windows if displayed.")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred in the example usage: {e}")
        import traceback
        traceback.print_exc()

    print("\nExample usage finished.")
```
