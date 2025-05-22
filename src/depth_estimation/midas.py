import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

class MiDaSDepthEstimator:
    """
    A class to estimate depth from an RGB image using a MiDaS model loaded from TensorFlow Hub.
    It handles model loading, image preprocessing, inference, and postprocessing.
    """
    def __init__(self, model_url: str = "https://tfhub.dev/intel/midas/v2_1/1"):
        """
        Initializes the MiDaS depth estimator by loading the model from TensorFlow Hub.

        Args:
            model_url (str, optional): The URL of the MiDaS model on TensorFlow Hub.
                                       Defaults to "https://tfhub.dev/intel/midas/v2_1/1" (a larger, more accurate model).
                                       Consider "https://tfhub.dev/intel/midas/v2_1_small/1" for a smaller, faster model.
        
        Raises:
            RuntimeError: If the model cannot be loaded from TensorFlow Hub.
        """
        self.model_url: str = model_url
        self.model: tf.keras.Model | None = None # For type hinting
        self.expected_input_height: int = 384 # Default, can be overridden by model signature
        self.expected_input_width: int = 384  # Default

        try:
            print(f"Loading MiDaS model from {self.model_url}...")
            self.model = hub.load(self.model_url)
            print("MiDaS model loaded successfully.")
            # Attempt to get input signature to determine expected input size
            try:
                # For some models, input_signature might be available directly
                # This is a common way, but might vary by specific model structure on TF Hub
                input_signature = self.model.signatures['serving_default']
                # The input tensor name can vary, common names are 'image', 'input_1', etc.
                # We are looking for the shape of the input tensor.
                # Example: TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32, name='image')
                # The shape can be (None, H, W, C) or (1, H, W, C)
                # Taking H and W from the shape.
                # This part is highly dependent on the specific model's signature.
                # For MiDaS v2.1, the models are dynamic in input size, but often have a preferred size like 384x384.
                # If a fixed size is needed and not easily found, we might default to a common size like 384.
                # For now, we'll assume MiDaS models are somewhat flexible or use a common default.
                # MiDaS v2.1 (large and small) are often used with 384x384 inputs.
                self.expected_input_height = 384
                self.expected_input_width = 384
                print(f"Model expects input of shape (approx): (None, {self.expected_input_height}, {self.expected_input_width}, 3)")
            except Exception as e:
                print(f"Could not automatically determine input size from model signature: {e}")
                print("Using default expected input size: 384x384.")
                self.expected_input_height = 384
                self.expected_input_width = 384

        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
            print("Please ensure TensorFlow and TensorFlow Hub are installed correctly,")
            print("and that the model URL is valid and accessible.")
            raise RuntimeError(f"Failed to load MiDaS model from {self.model_url}: {e}")

    def _preprocess_image(self, image_rgb: np.ndarray) -> tf.Tensor:
        """
        Preprocesses the input RGB image for the MiDaS model.
        This involves resizing to the model's expected input dimensions and normalizing pixel values.

        Args:
            image_rgb (numpy.ndarray): The input RGB image as a NumPy array (H, W, C),
                                       with pixel values in the range [0, 255].

        Returns:
            tf.Tensor: The preprocessed image as a TensorFlow tensor, ready for model inference.
                       Shape is (1, H_expected, W_expected, C).
        """
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")

        img_resized = cv2.resize(image_rgb, (self.expected_input_width, self.expected_input_height), 
                                 interpolation=cv2.INTER_CUBIC)
        # MiDaS models (TF Hub versions) typically expect input in the range [0,1] and of type float32
        img_normalized = img_resized.astype(np.float32) / 255.0
        # Expand dimensions to create a batch of 1 for the model: (1, H, W, C)
        img_tensor = tf.convert_to_tensor(img_normalized)[tf.newaxis, ...]
        return img_tensor

    def _postprocess_output(self, depth_tensor: tf.Tensor, 
                            original_height: int, original_width: int) -> np.ndarray:
        """
        Postprocesses the raw depth tensor output from the MiDaS model.
        This typically involves squeezing batch dimensions and resizing to the original image size.

        Args:
            depth_tensor (tf.Tensor): The raw output depth tensor from the model.
                                      MiDaS output is often relative inverse depth.
            original_height (int): The original height of the input image.
            original_width (int): The original width of the input image.

        Returns:
            numpy.ndarray: The postprocessed depth map, resized to original image dimensions (H_orig, W_orig).
                           The values are relative depth values.
        """
        # Squeeze batch and channel dimensions (MiDaS output is typically [1, H_model, W_model])
        depth_map_model_dims = tf.squeeze(depth_tensor).numpy()
        
        # Resize depth map to original image dimensions
        # INTER_CUBIC is a good general-purpose interpolation for resizing depth maps.
        # INTER_NEAREST could be used if preserving sharp discontinuities is critical and scale factor is integer,
        # but can lead to blockiness. INTER_LINEAR is faster but might blur details.
        depth_map_resized = cv2.resize(depth_map_model_dims, (original_width, original_height), 
                                       interpolation=cv2.INTER_CUBIC)
        return depth_map_resized

    def estimate_depth(self, image_rgb: np.ndarray, output_scale_factor: float = 1.0) -> np.ndarray:
        """
        Estimates depth from an RGB image using the loaded MiDaS model.

        The output depth map values are relative. The `output_scale_factor` can be used
        to scale these values to a pseudo-metric range, but finding the correct factor
        is scene-dependent and typically requires calibration or empirical tuning if
        absolute metric depth is needed.

        Args:
            image_rgb (numpy.ndarray): The input RGB image (H, W, C) with pixel values in [0, 255].
            output_scale_factor (float, optional): Factor to multiply the relative depth map by.
                                                   Useful for scaling to a pseudo-metric range.
                                                   Defaults to 1.0 (raw relative depth).

        Returns:
            numpy.ndarray: The estimated depth map, resized to original image dimensions (H_orig, W_orig)
                           and scaled by `output_scale_factor`.

        Raises:
            ValueError: If the input image is None or not in the expected format.
            RuntimeError: If model inference fails.
        """
        if image_rgb is None:
            raise ValueError("Input image cannot be None.")
        if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel RGB NumPy array (H, W, C).")

        original_height, original_width, _ = image_rgb.shape
        if self.model is None:
            raise RuntimeError("MiDaS model is not loaded. Cannot perform depth estimation.")

        preprocessed_image = self._preprocess_image(image_rgb)

        # Perform inference
        # The output key might vary, 'default' or specific names like 'midas_net/out/BiasAdd'
        # For TF Hub models, often it's just calling the model directly.
        try:
            prediction = self.model(preprocessed_image)
            # MiDaS TF Hub models usually output a dictionary. The key for the depth map can vary.
            # Common keys are 'output_1', 'depth', or the name of the output layer.
            # For intel/midas/v2_1/* models, the output is often directly the tensor,
            # or in a dict with a key like 'midas_v21_small_prediction' or 'midas_v21_prediction'
            # Let's try to access it assuming it's the first output or a known key.
            if isinstance(prediction, dict):
                # Check for common keys, this might need adjustment based on the exact model
                possible_keys = ['default', 'output_1', 'prediction', 'depth_map'] # Add more if known
                depth_tensor = None
                for key in possible_keys:
                    if key in prediction:
                        depth_tensor = prediction[key]
                        break
                if depth_tensor is None: # If no known key found, try to get the first value if it's a single-item dict
                    if len(prediction) == 1:
                        depth_tensor = next(iter(prediction.values()))
                    else:
                        raise RuntimeError(f"Could not find depth tensor in model output dictionary. Keys: {prediction.keys()}")
            else: # If the output is not a dict, assume it's the depth tensor directly
                depth_tensor = prediction
        except Exception as e:
            print(f"Error during model inference: {e}")
            # You might want to inspect `self.model.signatures` or the model documentation
            # on TF Hub to understand the exact output structure.
            print("Model output structure might be different than expected.")
            raise RuntimeError(f"Model inference failed: {e}")

        depth_map = self._postprocess_output(depth_tensor, original_height, original_width)
        
        # Apply the user-provided scaling factor
        scaled_depth_map = depth_map * output_scale_factor
        
        return scaled_depth_map

if __name__ == '__main__':
    # This block provides an example of how to use the MiDaSDepthEstimator.
    # It requires a GUI environment (like a desktop) to display images with OpenCV.
    print("MiDaSDepthEstimator - Example Usage")
    
    # Ensure TensorFlow uses the GPU if available and configured, otherwise CPU.
    # You can set TF_CPP_MIN_LOG_LEVEL to 2 to suppress TensorFlow INFO messages.
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    try:
        # Create a dummy RGB image (e.g., replace with an actual image loaded via cv2.imread)
        # Ensure the image is in RGB format if loaded with OpenCV (which loads as BGR).
        dummy_image_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        print(f"Created a dummy RGB image of shape: {dummy_image_rgb.shape}")

        # Initialize the estimator with the default (larger) MiDaS model
        # For a faster, smaller model, use:
        # estimator = MiDaSDepthEstimator(model_url="https://tfhub.dev/intel/midas/v2_1_small/1")
        estimator = MiDaSDepthEstimator() 

        print("Estimating depth (output is relative depth, scale factor 1.0)...")
        # The default output_scale_factor is 1.0, returning relative depth.
        relative_depth_map = estimator.estimate_depth(dummy_image_rgb) 
        print(f"Successfully estimated relative depth. Output map shape: {relative_depth_map.shape}")
        print(f"Relative depth map stats: min={np.min(relative_depth_map):.2f}, max={np.max(relative_depth_map):.2f}, mean={np.mean(relative_depth_map):.2f}")

        # Example of applying a scale factor for pseudo-metric depth.
        # The appropriate scale factor is scene-dependent.
        example_metric_scale = 10.0 
        print(f"Estimating depth (with example scale factor {example_metric_scale} for pseudo-metric output)...")
        pseudo_metric_depth_map = estimator.estimate_depth(dummy_image_rgb, output_scale_factor=example_metric_scale)
        print(f"Successfully estimated pseudo-metric depth. Output map shape: {pseudo_metric_depth_map.shape}")
        print(f"Pseudo-metric depth map stats: min={np.min(pseudo_metric_depth_map):.2f}, max={np.max(pseudo_metric_depth_map):.2f}, mean={np.mean(pseudo_metric_depth_map):.2f}")

        # Normalize the relative depth map for display purposes (0-255, 8-bit image)
        depth_display = cv2.normalize(relative_depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)

        print("Depth estimation complete. To view the results, you would typically use cv2.imshow().")
        # cv2.imshow("Dummy RGB", dummy_image_rgb)
        # cv2.imshow("Estimated Depth Colormap", depth_colormap)
        # print("Press any key to close windows if displayed.")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred in the example usage: {e}")
        import traceback
        traceback.print_exc()

    print("Example usage finished.")
