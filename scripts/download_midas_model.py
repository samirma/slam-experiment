import argparse
import sys
import os

# Adjust path to import from src
# This assumes the script is in 'scripts/' and 'src/' is a sibling directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.depth_estimation.midas import MiDaSDepthEstimator
except ImportError as e:
    print(f"Error importing MiDaSDepthEstimator: {e}")
    print("Please ensure the script is run from the project root (e.g., 'python scripts/download_midas_model.py')")
    print("or that 'src' directory is in your PYTHONPATH.")
    sys.exit(1)

MODELS_INFO = {
    "small": {"type_name": "MiDaS_small"},  # Standard small model (v2.1)
    "hybrid": {"type_name": "DPT_Hybrid"},   # DPT Hybrid model (v3.0 / v3.1 general)
    "large_beit_512": {"type_name": "DPT_BEiT_L_512"}, # DPT BEiT Large (v3.1)
    # Add other relevant v3.1 types if desired, e.g., "DPT_SwinV2_L_384"
}
# MODELS_BASE_DIR is no longer needed as PyTorch Hub manages its own cache.

def ensure_model_available(model_type_key: str):
    """
    Instantiates MiDaSDepthEstimator for the given model type key.
    This triggers download via torch.hub.load if the model is not already
    in the PyTorch cache.
    """
    if model_type_key not in MODELS_INFO:
        print(f"Error: Unknown model type key '{model_type_key}'. Cannot ensure availability.")
        return False

    info = MODELS_INFO[model_type_key]
    pytorch_model_name = info['type_name']
    
    print(f"\nEnsuring MiDaS model (PyTorch Hub type: '{pytorch_model_name}') is available...")
    
    success = False
    try:
        # Instantiating MiDaSDepthEstimator handles the actual download and setup via torch.hub.
        # The MiDaSDepthEstimator's __init__ method already prints detailed status.
        estimator = MiDaSDepthEstimator(midas_model_type=pytorch_model_name)
        
        # Check if the model was loaded successfully by the estimator
        if estimator.model is not None and estimator.transform is not None:
            # MiDaSDepthEstimator prints success messages, so we just confirm here.
            print(f"MiDaS model '{pytorch_model_name}' seems ready (instantiation successful).")
            success = True
        else:
            # This case might be reached if MiDaSDepthEstimator's init has an issue
            # but doesn't raise an exception that's caught below (should be rare).
            print(f"MiDaS model '{pytorch_model_name}' could not be fully prepared by MiDaSDepthEstimator. "
                  "Check previous messages from the estimator for details.")
            success = False
            
    except RuntimeError as e: # Catch errors from MiDaSDepthEstimator's init
        print(f"Error during setup for MiDaS model '{pytorch_model_name}': {e}")
        print("This could be due to network issues, an invalid model name for torch.hub, or other PyTorch Hub errors.")
        success = False
    except ImportError as e:
        print(f"ImportError during setup for MiDaS model '{pytorch_model_name}': {e}")
        print("Please ensure PyTorch and all necessary dependencies for MiDaS are installed.")
        success = False
    except Exception as e:
        print(f"An unexpected error occurred while ensuring MiDaS model '{pytorch_model_name}' availability: {e}")
        import traceback
        traceback.print_exc()
        success = False
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensures specified MiDaS depth estimation models (PyTorch versions) are downloaded by PyTorch Hub.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=list(MODELS_INFO.keys()) + ['all'], 
        default='hybrid', 
        help=(
            "Type of MiDaS model to ensure is downloaded.\n"
            f"Choices: {', '.join(list(MODELS_INFO.keys()) + ['all'])}.\n"
            "'small': MiDaS v2.1 Small model.\n"
            "'hybrid': DPT Hybrid model (versatile v3.0/v3.1).\n"
            "'large_beit_512': DPT BEiT Large model (v3.1, high accuracy).\n"
            "'all': Ensure all defined models are downloaded.\n"
            "Default is 'hybrid'."
        )
    )
    args = parser.parse_args()

    print(f"Selected model type to ensure: {args.model_type}")

    all_successful = True
    if args.model_type == 'all':
        print("\nEnsuring all defined MiDaS models are available...")
        for key in MODELS_INFO:
            if not ensure_model_available(key):
                all_successful = False
                print(f"Failed to ensure model for key '{key}' (PyTorch Hub type: '{MODELS_INFO[key]['type_name']}')")
    else:
        if not ensure_model_available(args.model_type):
            all_successful = False
    
    if all_successful:
        print("\nAll requested MiDaS models appear to be available in the PyTorch Hub cache.")
    else:
        print("\nSome MiDaS models could not be prepared. Please check the log messages above.")
        sys.exit(1) # Exit with error code if any download failed

    print("Model download process finished.")
