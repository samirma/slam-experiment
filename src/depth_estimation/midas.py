import torch
import cv2
import numpy as np
import os # Still used for path operations if any, but not for model loading directly

class MiDaSDepthEstimator:
    """
    A class to estimate depth from an RGB image using a MiDaS model from PyTorch Hub.
    It handles model loading, image preprocessing (transformation), inference, and postprocessing.
    """
    def __init__(self, midas_model_type: str = "DPT_Hybrid"):
        """
        Initializes the MiDaS depth estimator using PyTorch Hub.

        Args:
            midas_model_type (str, optional): The type of MiDaS model to load.
                                              Examples: "DPT_Hybrid", "MiDaS_small", "DPT_BEiT_L_512".
                                              Defaults to "DPT_Hybrid".
        
        Raises:
            RuntimeError: If the model or transform cannot be loaded from PyTorch Hub.
            ImportError: If PyTorch or other essential libraries are not installed.
        """
        self.midas_model_type = midas_model_type
        self.model = None
        self.transform = None
        self.device = None

        try:
            print(f"Attempting to load MiDaS model '{self.midas_model_type}' from PyTorch Hub...")
            # Load the model
            self.model = torch.hub.load("intel-isl/MiDaS", self.midas_model_type, trust_repo=True)
            
            # Determine and load the appropriate transform
            # Based on typical naming conventions from intel-isl/MiDaS or general transforms.
            transform_name_candidate = self.midas_model_type + "_transform" 
            try:
                print(f"Attempting to load specific transform: '{transform_name_candidate}'...")
                self.transform = torch.hub.load("intel-isl/MiDaS", transform_name_candidate, trust_repo=True)
                print(f"Successfully loaded transform: '{transform_name_candidate}'.")
            except Exception:
                print(f"Could not load specific transform '{transform_name_candidate}'. Falling back to general transforms.")
                if "dpt" in self.midas_model_type.lower() or \
                   "beit" in self.midas_model_type.lower() or \
                   "swin" in self.midas_model_type.lower():
                    print("Using 'dpt_transform' as fallback.")
                    self.transform = torch.hub.load("intel-isl/MiDaS", "dpt_transform", trust_repo=True)
                else: # For older/simpler models like MiDaS_small if it's not DPT based
                    print("Using 'midas_transform' as fallback.")
                    self.transform = torch.hub.load("intel-isl/MiDaS", "midas_transform", trust_repo=True)
                print("Fallback transform loaded successfully.")

            # Set device and move model to device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode

            print(f"MiDaS model '{self.midas_model_type}' and its transform loaded to {self.device} successfully.")

        except Exception as e:
            print(f"Error initializing MiDaS PyTorch model '{self.midas_model_type}': {e}")
            import traceback
            traceback.print_exc()
            # Propagate error to indicate failure
            raise RuntimeError(f"Failed to initialize MiDaS PyTorch model '{self.midas_model_type}': {e}")


    def _transform_input(self, image_bgr: np.ndarray) -> torch.Tensor:
        """
        Preprocesses the input BGR image using the loaded MiDaS transform.
        Converts BGR to RGB, applies model-specific transformations (resize, normalize, etc.),
        and moves the tensor to the configured device.

        Args:
            image_bgr (np.ndarray): Input image in BGR format (H, W, C).

        Returns:
            torch.Tensor: The transformed image tensor, ready for model input.
        
        Raises:
            TypeError: If input is not a NumPy array.
            ValueError: If input is not a 3-channel image.
        """
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel BGR NumPy array (H, W, C).")

        # Convert BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply the MiDaS-specific transform
        # The transform typically handles resizing, normalization, and conversion to tensor
        input_tensor = self.transform(image_rgb).to(self.device)
        return input_tensor

    def _postprocess_output(self, prediction_tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocesses the raw model output tensor.
        Moves tensor to CPU, converts to NumPy array, and removes batch dimension.
        Note: Resizing to original image dimensions is handled in `estimate_depth`
        after interpolation, as per common PyTorch MiDaS examples.

        Args:
            prediction_tensor (torch.Tensor): Raw output tensor from the model,
                                              after potential interpolation to original image size.

        Returns:
            np.ndarray: Depth map as a NumPy array.
        """
        depth_map_numpy = prediction_tensor.cpu().numpy()
        # Squeeze is not needed here if interpolation already produced (H, W)
        return depth_map_numpy

    def estimate_depth(self, image_bgr: np.ndarray, output_scale_factor: float = 1.0) -> np.ndarray:
        """
        Estimates depth from a BGR image.

        Args:
            image_bgr (np.ndarray): Input image in BGR format (H, W, C).
            output_scale_factor (float, optional): A factor to scale the output depth map.
                                                   Useful if the raw output needs adjustment.
                                                   Defaults to 1.0.

        Returns:
            np.ndarray: The estimated depth map, scaled by `output_scale_factor`.
        
        Raises:
            RuntimeError: If the model is not loaded or if inference fails.
            ValueError: If input image is invalid.
        """
        if image_bgr is None:
            raise ValueError("Input image cannot be None.")
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel BGR NumPy array (H, W, C).")

        if self.model is None or self.transform is None:
            raise RuntimeError("MiDaS PyTorch model or transform is not loaded. Cannot perform depth estimation.")

        original_height, original_width, _ = image_bgr.shape
        
        # Preprocess the image (includes BGR to RGB, transform, and to_device)
        transformed_input = self._transform_input(image_bgr)

        try:
            with torch.no_grad(): # Disable gradient calculations for inference
                prediction = self.model(transformed_input)

            # Interpolate prediction to original image size
            # MiDaS PyTorch models often output a tensor that needs to be resized.
            # The output is typically [1, H_model, W_model]. interpolate needs [N, C, H, W].
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), # Add channel dimension: [1, 1, H_model, W_model]
                size=(original_height, original_width),
                mode="bicubic", # "bicubic" or "bilinear" are common
                align_corners=False, # Usually False for MiDaS, check model specifics if issues
            ).squeeze() # Remove batch and channel: [H_orig, W_orig]

        except Exception as e:
            print(f"Error during MiDaS PyTorch model inference: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"MiDaS PyTorch model inference failed: {e}")

        # Postprocess the output (CPU, NumPy)
        # Resizing is now handled by the interpolate step above.
        depth_map = self._postprocess_output(prediction)
        
        # Apply scale factor
        scaled_depth_map = depth_map * output_scale_factor
        return scaled_depth_map

if __name__ == '__main__':
    print("MiDaSDepthEstimator (PyTorch) - Example Usage")
    
    # Create a dummy BGR image (OpenCV default format)
    dummy_image_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"Created a dummy BGR image of shape: {dummy_image_bgr.shape}")

    model_types_to_test = ["DPT_Hybrid", "MiDaS_small"] 
    # Note: "MiDaS_small" is a v2.1 model type but often works with torch.hub.
    # For v3.1 specific small models, one might need to check intel-isl/MiDaS repo for exact names.

    for model_type in model_types_to_test:
        print(f"\n--- Testing MiDaS Model Type: {model_type} ---")
        try:
            print(f"Initializing MiDaSDepthEstimator with model type: {model_type}...")
            estimator = MiDaSDepthEstimator(midas_model_type=model_type)
            
            print(f"Estimating depth for dummy image with {model_type}...")
            # Pass the BGR image directly
            relative_depth_map = estimator.estimate_depth(dummy_image_bgr)
            
            print(f"{model_type} - Relative depth map stats: "
                  f"min={np.min(relative_depth_map):.2f}, max={np.max(relative_depth_map):.2f}, "
                  f"mean={np.mean(relative_depth_map):.2f}, shape={relative_depth_map.shape}")

            # Optional: Display results if a GUI is available and cv2 is configured
            # This requires a desktop environment.
            # depth_display = cv2.normalize(relative_depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)
            # cv2.imshow(f"Dummy BGR (Input to {model_type})", dummy_image_bgr)
            # cv2.imshow(f"Depth ({model_type})", depth_colormap)
            # print(f"Displaying depth for {model_type}. Press any key in an OpenCV window to continue to next model or exit...")
            # cv2.waitKey(0)

        except RuntimeError as e:
            print(f"RuntimeError for model type {model_type}: {e}")
            print("This might be due to model download issues, unsupported model type by torch.hub, or resource limitations.")
        except ImportError as e:
            print(f"ImportError for model type {model_type}: {e}. Please ensure PyTorch and torchvision are installed.")
        except Exception as e:
            print(f"An unexpected error occurred for model type {model_type}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # cv2.destroyAllWindows() # Close any OpenCV windows for this model type test
            pass
            
    # cv2.destroyAllWindows() # Final cleanup of any remaining windows
    print("\nMiDaS PyTorch example usage finished.")
