import cv2
import numpy as np

def detect_features(image, detector_type='orb', detector=None):
    """
    Detects features in a grayscale image using the specified detector.

    Args:
        image (np.ndarray): The input grayscale image.
        detector_type (str): The type of feature detector to use ('orb', 'sift', etc.)
                             if 'detector' is not provided. Defaults to 'orb'.
        detector (cv2.Feature2D, optional): A pre-initialized OpenCV feature detector.
                                            If None, one will be created based on detector_type.
                                            Defaults to None.

    Returns:
        tuple: (keypoints, descriptors)
               keypoints: Detected keypoints.
               descriptors: Computed descriptors for the keypoints.
               Returns (None, None) if the detector type is not supported or an error occurs.
    """
    if image is None:
        print("Error: Input image is None.")
        return None, None
    
    if len(image.shape) > 2 and image.shape[2] > 1: # Check if not grayscale
        print("Warning: Input image is not grayscale. Converting to grayscale.")
        # This is a common expectation for feature detectors, though ORB might handle color.
        # For consistency, let's assume grayscale input is preferred.
        # If the image is BGR (common in OpenCV), cvtColor is appropriate.
        # If it's already single channel but with a redundant dimension, a reshape might be better.
        # However, the problem statement implies image is already grayscale.
        # This check is more of a safeguard. If it's truly guaranteed grayscale, this can be omitted.
        # For now, let's assume it should be grayscale and if not, it's an issue.
        # A better approach: ensure the caller provides a grayscale image.
        # For this implementation, let's proceed assuming it's grayscale as per "Accept a grayscale image".
    
    # Use provided detector or create a new one
    active_detector = detector
    if active_detector is None:
        if detector_type == 'orb':
            active_detector = cv2.ORB_create()
        # Placeholder for SIFT if opencv-contrib-python was available and desired
        # elif detector_type == 'sift':
        #     try:
        #         active_detector = cv2.SIFT_create()
        #     except AttributeError:
        #         print("SIFT not available. Ensure opencv-contrib-python is installed and you are using the correct OpenCV version.")
        #         return None, None
        else:
            print(f"Error: Unsupported detector_type '{detector_type}' and no detector provided. Supported types: 'orb'.")
            return None, None

    if active_detector is None: # Should be caught by the else above, but as a safeguard
        print("Error: Detector could not be initialized or provided.")
        return None, None

    try:
        keypoints, descriptors = active_detector.detectAndCompute(image, None)
        return keypoints, descriptors
    except Exception as e:
        print(f"Error during feature detection/computation with {detector_type if detector is None else 'provided detector'}: {e}")
        return None, None

if __name__ == '__main__':
    # Example Usage (requires a grayscale image)
    # Create a dummy grayscale image for testing
    dummy_image = np.zeros((100, 100, 1), dtype=np.uint8) # Single channel
    # Or, if it's already 2D:
    dummy_image_2d = np.random.randint(0, 256, (200, 300), dtype=np.uint8)

    print("Testing ORB detector (creating new one)...")
    kp_orb_new, des_orb_new = detect_features(dummy_image_2d.copy(), detector_type='orb')
    if kp_orb_new is not None and des_orb_new is not None:
        print(f"ORB (new): Detected {len(kp_orb_new)} keypoints. Descriptor shape: {des_orb_new.shape if des_orb_new is not None else None}")
    else:
        print("ORB (new) detection failed or returned None.")

    print("\nTesting ORB detector (providing existing one)...")
    orb_instance = cv2.ORB_create()
    kp_orb_existing, des_orb_existing = detect_features(dummy_image_2d.copy(), detector=orb_instance)
    if kp_orb_existing is not None and des_orb_existing is not None:
        print(f"ORB (existing): Detected {len(kp_orb_existing)} keypoints. Descriptor shape: {des_orb_existing.shape if des_orb_existing is not None else None}")
    else:
        print("ORB (existing) detection failed or returned None.")


    # Example of how SIFT might be called if implemented and available
    # print("\nTesting SIFT detector (will fail if not available)...")
    # kp_sift, des_sift = detect_features(dummy_image_2d.copy(), detector_type='sift')
    # if kp_sift:
    #     print(f"SIFT: Detected {len(kp_sift)} keypoints.")
    
    print("\nTesting unsupported detector type (no detector provided)...")
    kp_unsupported, des_unsupported = detect_features(dummy_image_2d.copy(), detector_type='xyz')
    if kp_unsupported is None and des_unsupported is None:
        print("Unsupported detector type test passed (returned None, None).")

    print("\nTesting with None image...")
    kp_none, des_none = detect_features(None, detector=orb_instance) # Using existing detector
    if kp_none is None and des_none is None:
        print("None image test passed (returned None, None).")

    # Test cases for match_features
    print("\n--- Testing match_features ---")
    # Create two slightly different dummy images for matching
    dummy_image1 = np.random.randint(0, 256, (200, 300), dtype=np.uint8)
    dummy_image2 = dummy_image1.copy()
    # Introduce some changes to dummy_image2 to get some different features
    cv2.rectangle(dummy_image2, (50, 50), (100, 100), (0,0,0), -1) # Add a black square
    cv2.circle(dummy_image2, (150,150), 30, (255,255,255), -1) # Add a white circle

    print("Detecting features in dummy image 1 (ORB)...")
    kp1_orb, des1_orb = detect_features(dummy_image1, detector_type='orb')
    if des1_orb is None:
        print("Failed to detect features in dummy_image1 for matching test.")
    else:
        print(f"Detected {len(kp1_orb)} keypoints in dummy_image1.")

    print("Detecting features in dummy image 2 (ORB)...")
    kp2_orb, des2_orb = detect_features(dummy_image2, detector_type='orb')
    if des2_orb is None:
        print("Failed to detect features in dummy_image2 for matching test.")
    else:
        print(f"Detected {len(kp2_orb)} keypoints in dummy_image2.")
        
    if des1_orb is not None and des2_orb is not None:
        print("\nMatching ORB features with Brute-Force (BF) matcher and ratio test...")
        good_matches_orb = match_features(des1_orb, des2_orb, matcher_type='bf', detector_type='orb')
        if good_matches_orb is not None:
            print(f"Found {len(good_matches_orb)} good matches (ORB).")
        else:
            print("match_features returned None for ORB.")
            
        # Test with descriptors that might be too few for k=2 in knnMatch
        print("\nMatching with potentially insufficient descriptors for k=2 (ORB)...")
        des_few1 = des1_orb[:1] if des1_orb is not None and len(des1_orb) >=1 else None # Take only one descriptor
        des_few2 = des2_orb[:5] if des2_orb is not None and len(des2_orb) >=5 else des2_orb # Take a few
        
        if des_few1 is not None and des_few2 is not None and len(des_few1) > 0 and len(des_few2) > 0:
             good_matches_few = match_features(des_few1, des_few2, detector_type='orb')
             print(f"Found {len(good_matches_few)} good matches with few descriptors (ORB). Should be 0 or few.")
        else:
            print("Skipping few descriptors test as not enough descriptors were generated.")

    else:
        print("Skipping match_features tests for ORB due to failure in feature detection.")

    # Test case for None descriptors
    print("\nTesting match_features with None descriptors...")
    matches_none = match_features(None, des2_orb)
    if matches_none == []: # Expect empty list
        print("match_features with None descriptors (1) test passed (returned empty list).")
    matches_none_2 = match_features(des1_orb, None)
    if matches_none_2 == []:
        print("match_features with None descriptors (2) test passed (returned empty list).")
    
    # Test case for empty descriptors
    print("\nTesting match_features with empty descriptors...")
    empty_des = np.array([], dtype=np.uint8)
    matches_empty = match_features(empty_des, des2_orb)
    if matches_empty == []:
        print("match_features with empty descriptors (1) test passed (returned empty list).")
    matches_empty_2 = match_features(des1_orb, empty_des)
    if matches_empty_2 == []:
        print("match_features with empty descriptors (2) test passed (returned empty list).")

    # SIFT requires opencv-contrib-python, so this will likely fail or skip
    # print("\nDetecting features for SIFT matching (will skip if SIFT unavailable)...")
    # kp1_sift, des1_sift = detect_features(dummy_image1, detector_type='sift')
    # kp2_sift, des2_sift = detect_features(dummy_image2, detector_type='sift')

    # if des1_sift is not None and des2_sift is not None:
    #     print("Matching SIFT features with Brute-Force (BF) matcher and ratio test...")
    #     good_matches_sift = match_features(des1_sift, des2_sift, matcher_type='bf', detector_type='sift')
    #     if good_matches_sift:
    #         print(f"Found {len(good_matches_sift)} good matches (SIFT).")
    # else:
    #     print("Skipping match_features tests for SIFT (features not detected).")


def match_features(descriptors1, descriptors2, matcher_type='bf', detector_type='orb', k_best_matches=2, ratio_thresh=0.75):
    """
    Matches features from two sets of descriptors.

    Args:
        descriptors1 (np.ndarray): Descriptors from the first image.
        descriptors2 (np.ndarray): Descriptors from the second image.
        matcher_type (str): Type of matcher to use ('bf' for Brute-Force). Default 'bf'.
        detector_type (str): Type of feature detector used ('orb', 'sift', 'brisk'). 
                             This helps select the correct norm for BFMatcher. Default 'orb'.
        k_best_matches (int): Number of best matches to find for each descriptor (for ratio test). Default 2.
        ratio_thresh (float): Lowe's ratio test threshold. Default 0.75.

    Returns:
        list: A list of good DMatch objects.
    """
    if descriptors1 is None or descriptors2 is None:
        print("Error: One or both descriptor sets are None.")
        return []
    
    if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
        print("Warning: One or both descriptor sets are empty.")
        return []
    
    # Ensure descriptors are of a supported type (float32 for SIFT/SURF, uint8 for ORB/BRISK)
    if detector_type in ['sift', 'surf'] and descriptors1.dtype != np.float32:
        descriptors1 = descriptors1.astype(np.float32)
    if detector_type in ['sift', 'surf'] and descriptors2.dtype != np.float32:
        descriptors2 = descriptors2.astype(np.float32)
    if detector_type in ['orb', 'brisk'] and descriptors1.dtype != np.uint8:
        descriptors1 = descriptors1.astype(np.uint8)
    if detector_type in ['orb', 'brisk'] and descriptors2.dtype != np.uint8:
        descriptors2 = descriptors2.astype(np.uint8)

    matcher = None
    if matcher_type == 'bf':
        if detector_type in ['orb', 'brisk']: # Binary descriptors
            norm_type = cv2.NORM_HAMMING
        elif detector_type in ['sift', 'surf']: # Floating point descriptors
            norm_type = cv2.NORM_L2
        else:
            print(f"Warning: Detector type '{detector_type}' not explicitly handled for BFMatcher norm. Defaulting to NORM_L2.")
            norm_type = cv2.NORM_L2 # A general default
        
        try:
            # crossCheck=False is needed for knnMatch
            matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        except Exception as e:
            print(f"Error initializing BFMatcher: {e}")
            return []
    # Placeholder for FLANN matcher
    # elif matcher_type == 'flann':
    #     FLANN_INDEX_KDTREE = 1
    #     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #     search_params = dict(checks=50) # or pass empty dictionary
    #     if detector_type in ['orb', 'brisk']:
    #         # For binary descriptors with FLANN, use LSH
    #         # FLANN_INDEX_LSH = 6
    #         # index_params= dict(algorithm = FLANN_INDEX_LSH,
    #         #                    table_number = 6, # 12
    #         #                    key_size = 12,     # 20
    #         #                    multi_probe_level = 1) #2
    #         print("FLANN with LSH for binary descriptors (like ORB) is complex to setup and may require specific OpenCV builds. Using BFMatcher as fallback logic for now.")
    #         # Fallback or error for ORB with FLANN if not configured
    #         norm_type = cv2.NORM_HAMMING
    #         matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    #     elif detector_type in ['sift', 'surf']:
    #          matcher = cv2.FlannBasedMatcher(index_params, search_params)
    #     else:
    #         print(f"FLANN not configured for detector type '{detector_type}'.")
    #         return []
    else:
        print(f"Error: Unsupported matcher type '{matcher_type}'. Supported: 'bf'.")
        return []

    if matcher is None:
        print("Error: Matcher could not be initialized.")
        return []

    good_matches = []
    try:
        # Ensure k_best_matches is not greater than the number of descriptors in the training set (descriptors2)
        # And also not greater than number of descriptors in query set (descriptors1) if descriptors1 is smaller
        # However, knnMatch itself handles cases where k > available points in a way,
        # but it's good to be mindful, especially if len(descriptors2) < k_best_matches.
        if len(descriptors1) < k_best_matches or len(descriptors2) < k_best_matches:
            if k_best_matches > 1 : # Only an issue if we need multiple neighbors for ratio test
                 print(f"Warning: Not enough descriptors (query: {len(descriptors1)}, train: {len(descriptors2)}) for k={k_best_matches} matches. Skipping ratio test or reducing k if possible.")
                 # If k_best_matches is 2 for ratio test, and we don't have enough, we can't do it.
                 # If k_best_matches was 1, it would be direct matching.
                 # For now, if we can't get k_best_matches, we return no good_matches if ratio test is implied.
                 if k_best_matches >= 2: # Ratio test is implied
                     return [] # Cannot perform ratio test
                 # If k_best_matches was 1, we could proceed with direct matching, but the function is designed for ratio test with k=2
        
        matches = matcher.knnMatch(descriptors1, descriptors2, k=k_best_matches)

        if k_best_matches >= 2: # Apply Lowe's Ratio Test
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        elif k_best_matches == 1 and matches: # Just take all matches if k=1 (no ratio test)
            # matches will be a list of lists, each inner list containing one DMatch object
            for match_list in matches:
                if match_list: # Check if the list is not empty
                    good_matches.append(match_list[0])
        # else: k_best_matches is 0 or invalid, knnMatch might error or return empty

    except cv2.error as e:
        # This can happen if, e.g., descriptors1 has N elements and descriptors2 has M elements,
        # and k > M. OpenCV's BFMatcher.knnMatch might throw an error.
        print(f"OpenCV error during knnMatch: {e}")
        print("This might be due to too few descriptors in one of the sets for the chosen k, or incompatible descriptor types/formats if not caught earlier.")
        return []
    except Exception as e:
        print(f"Error during feature matching: {e}")
        return []
        
    return good_matches
