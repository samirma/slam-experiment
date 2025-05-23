import os
import requests
import tarfile
import shutil
from tqdm import tqdm
from pathlib import Path # Not strictly used in this version, but good for future path ops

def download_and_extract_tar_gz(
    model_url: str,
    target_extraction_dir: str,
    archive_filename_suggestion: str = "model.tar.gz"
) -> str:
    """
    Downloads a tar.gz archive from a URL, displays a progress bar,
    validates the downloaded content (Content-Type and gzip magic number),
    extracts it to a target directory, and cleans up the archive.

    Args:
        model_url (str): The URL from which to download the tar.gz archive.
        target_extraction_dir (str): The directory where the archive contents
                                     will be extracted. This directory will be
                                     created if it doesn't exist.
        archive_filename_suggestion (str, optional): A suggested filename for the
                                                     downloaded archive if one cannot
                                                     be inferred from the URL.
                                                     Defaults to "model.tar.gz".

    Returns:
        str: The path to the `target_extraction_dir` where files were extracted.

    Raises:
        RuntimeError: If any step (download, validation, extraction) fails.
                      The error message will contain details about the failure.
        requests.exceptions.RequestException: For network-related issues during download.
    """
    print(f"Ensuring target directory exists: {target_extraction_dir}")
    os.makedirs(target_extraction_dir, exist_ok=True)

    # Determine archive filename
    try:
        # Try to get filename from URL first
        url_path = model_url.split("?")[0] # Remove query parameters
        archive_filename = os.path.basename(url_path)
        if not (archive_filename.endswith(".tar.gz") or archive_filename.endswith(".tgz")):
            print(f"URL did not yield a .tar.gz/.tgz filename ('{archive_filename}'), using suggestion: '{archive_filename_suggestion}'")
            archive_filename = archive_filename_suggestion
    except Exception: # Fallback if URL parsing is tricky
        archive_filename = archive_filename_suggestion
        print(f"Could not reliably determine filename from URL, using suggestion: '{archive_filename}'")

    temp_archive_path = os.path.join(target_extraction_dir, archive_filename)

    try:
        print(f"Attempting to download archive from {model_url} to {temp_archive_path}...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()  # Check for HTTP errors (4xx or 5xx)

        # Robust Download Check 1: Content-Type
        content_type = response.headers.get('Content-Type', '').lower()
        expected_archive_content_types = [
            'application/gzip', 'application/x-gzip',
            'application/x-tar', 'application/octet-stream', # Common for binary files
            'application/x-compressed-tar', # Another possibility
        ]
        print(f"Received Content-Type: {content_type}")
        if 'text/html' in content_type and not any(ct in content_type for ct in expected_archive_content_types):
            # Attempt to get more info from the response if it's small (e.g. an error page)
            error_page_snippet = ""
            try:
                # Read a small part of the response content if it's likely text
                response_content_sample = response.text[:500] if response.content else ""
                error_page_snippet = f" Response snippet (if HTML error): '{response_content_sample.strip()[:200]}...'"
            except Exception: # Don't let snippet generation fail the main error
                pass

            raise RuntimeError(
                f"Download failed: Expected an archive, but received content type '{content_type}'. "
                f"This might be an HTML error page from the server.{error_page_snippet} Please check the model URL."
            )

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 8192  # Standard block size

        print(f"Starting download of {archive_filename} ({total_size_in_bytes / (1024*1024):.2f} MB)...")
        with open(temp_archive_path, 'wb') as f, tqdm(
            desc=f"Downloading {archive_filename}",
            total=total_size_in_bytes,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave=False
        ) as bar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:  # filter out keep-alive new chunks
                    size = f.write(chunk)
                    bar.update(size)
        
        if total_size_in_bytes != 0 and bar.n != total_size_in_bytes:
            # This might happen if connection is interrupted or server doesn't send full file
            # For now, it's a warning, but could be an error.
            print(f"Warning: Downloaded size ({bar.n} bytes) does not match Content-Length ({total_size_in_bytes} bytes). File might be incomplete.")
        print("Download complete.")

        # Robust Download Check 2: Gzip magic number
        print(f"Verifying integrity of downloaded archive: {temp_archive_path}...")
        try:
            with open(temp_archive_path, 'rb') as f_check:
                file_start_bytes = f_check.read(2)
            if file_start_bytes != b'\x1f\x8b':
                additional_info = ""
                try:
                    with open(temp_archive_path, 'r', encoding='utf-8', errors='ignore') as f_text_check:
                        additional_info = f_text_check.read(200)
                except Exception: # Ignore if reading as text fails
                    pass
                raise RuntimeError(
                    f"Downloaded file at {temp_archive_path} is not a valid gzip archive. "
                    f"Magic bytes mismatch. Expected b'\\x1f\\x8b', got {file_start_bytes}. "
                    f"Start of file (if text): '{additional_info.strip()[:100]}...'"
                )
            print("Archive integrity (magic number) verified.")
        except Exception as e:
            raise RuntimeError(f"Could not verify downloaded file format at {temp_archive_path}: {e}")

        print(f"Extracting archive {temp_archive_path} to {target_extraction_dir}...")
        with tarfile.open(temp_archive_path, "r:gz") as tar:
            tar.extractall(path=target_extraction_dir)
        print("Extraction complete.")
        
        return target_extraction_dir

    except requests.exceptions.RequestException as e:
        # More specific error for network/HTTP issues
        raise RuntimeError(f"Network error or HTTP error during download from {model_url}: {e}")
    except tarfile.TarError as e:
        raise RuntimeError(f"Failed to extract model archive {temp_archive_path}: {e}. The file might be corrupted.")
    except IOError as e: # Catches file open/read/write issues
        raise RuntimeError(f"File I/O error during download/extraction process for {temp_archive_path}: {e}")
    except Exception as e:
        # Catch-all for other unexpected errors
        # Ensure temp_archive_path is defined before trying to use it in the error message
        archive_path_msg_part = f" (archive path: {temp_archive_path})" if 'temp_archive_path' in locals() else ""
        raise RuntimeError(f"An unexpected error occurred during model download/extraction{archive_path_msg_part}: {e}")
    finally:
        # Cleanup the temporary archive file if it exists
        if os.path.exists(temp_archive_path) and os.path.isfile(temp_archive_path):
            try:
                os.remove(temp_archive_path)
                print(f"Successfully removed temporary archive: {temp_archive_path}")
            except OSError as rm_err:
                print(f"Warning: Could not remove temporary archive {temp_archive_path}: {rm_err}")

if __name__ == '__main__':
    print("Testing model_downloader.py...")

    # --- Example Usage ---
    # Create a dummy tar.gz file for testing locally
    test_base_dir = Path("temp_downloader_test")
    test_base_dir.mkdir(parents=True, exist_ok=True)
    
    dummy_archive_name = "test_model.tar.gz"
    dummy_archive_path = test_base_dir / dummy_archive_name
    dummy_content_dir = test_base_dir / "dummy_content"
    dummy_content_dir.mkdir(exist_ok=True)
    
    # Create some dummy files to include in the archive
    with open(dummy_content_dir / "file1.txt", "w") as f:
        f.write("This is file1.")
    with open(dummy_content_dir / "file2.txt", "w") as f:
        f.write("This is file2.")
    
    # Create the tar.gz archive
    print(f"Creating dummy archive at {dummy_archive_path} for testing...")
    with tarfile.open(dummy_archive_path, "w:gz") as tar:
        # Add files from dummy_content_dir to the root of the archive
        tar.add(dummy_content_dir / "file1.txt", arcname="file1.txt")
        tar.add(dummy_content_dir / "file2.txt", arcname="file2.txt")
        # Optionally, add a versioned subdirectory like TF Hub models often have
        versioned_subdir_in_archive = "v1"
        tar.add(dummy_content_dir / "file1.txt", arcname=os.path.join(versioned_subdir_in_archive, "model_file.txt"))
    print("Dummy archive created.")

    # Use file:// URL for local testing
    # Convert WindowsPath to a valid file URL if on Windows
    if os.name == 'nt':
        dummy_model_url = f"file:///{str(dummy_archive_path.resolve()).replace(os.sep, '/')}"
    else:
        dummy_model_url = f"file://{str(dummy_archive_path.resolve())}"
        
    target_dir = str(test_base_dir / "extracted_model")

    print(f"\nAttempting to download and extract from local dummy URL: {dummy_model_url}")
    print(f"Target extraction directory: {target_dir}")

    try:
        extracted_path = download_and_extract_tar_gz(
            model_url=dummy_model_url,
            target_extraction_dir=target_dir,
            archive_filename_suggestion="suggested_name.tar.gz" # Test suggestion if URL parsing fails
        )
        print(f"\nDownload and extraction successful. Extracted to: {extracted_path}")
        print("Contents of extraction directory:")
        for item in os.listdir(extracted_path):
            item_path = os.path.join(extracted_path, item)
            if os.path.isdir(item_path):
                print(f"  - {item}/")
                for sub_item in os.listdir(item_path):
                    print(f"    - {sub_item}")
            else:
                print(f"  - {item}")
        
        # Verify specific files
        assert os.path.exists(os.path.join(extracted_path, "file1.txt")), "file1.txt not found"
        assert os.path.exists(os.path.join(extracted_path, versioned_subdir_in_archive, "model_file.txt")), "model_file.txt in subdir not found"
        print("\nKey files verified successfully.")

    except RuntimeError as e:
        print(f"Error during test: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up test directory...")
        try:
            shutil.rmtree(test_base_dir)
            print(f"Removed test directory: {test_base_dir}")
        except OSError as e:
            print(f"Error cleaning up test directory {test_base_dir}: {e}")

    print("\n--- Test with a non-existent URL (expecting failure) ---")
    non_existent_url = "http://localhost/non_existent_model.tar.gz" # Assuming nothing runs on localhost
    target_dir_fail_test = str(test_base_dir / "extracted_model_fail")
    try:
        download_and_extract_tar_gz(non_existent_url, target_dir_fail_test)
    except RuntimeError as e:
        print(f"Successfully caught expected error for non-existent URL: {e}")
    except Exception as e:
        print(f"Caught an unexpected error type for non-existent URL: {e}")
    finally:
        if os.path.exists(target_dir_fail_test):
            shutil.rmtree(target_dir_fail_test)

    print("\nmodel_downloader.py test finished.")
    # Note: To test with real URLs, replace dummy_model_url.
    # Example: Real TF Hub model URL (small MiDaS)
    # real_model_url = "https://tfhub.dev/intel/midas/v2_1_small/1?tf-hub-format=compressed"
    # real_target_dir = "models_test_download/midas_v2_1_tfhub_small_real"
    # try:
    #     print(f"\n--- Testing with REAL model URL: {real_model_url} ---")
    #     extracted_path_real = download_and_extract_tar_gz(real_model_url, real_target_dir)
    #     print(f"Successfully downloaded and extracted real model to: {extracted_path_real}")
    #     # Check for saved_model.pb or other expected files
    #     # Example: find saved_model.pb (can be nested)
    #     found_pb = False
    #     for root, dirs, files in os.walk(extracted_path_real):
    #         if "saved_model.pb" in files:
    #             print(f"Found 'saved_model.pb' in {root}")
    #             found_pb = True
    #             break
    #     assert found_pb, "'saved_model.pb' not found in extracted real model."
    # except RuntimeError as e:
    #     print(f"Error downloading real model: {e}")
    # except Exception as e:
    #     print(f"Unexpected error with real model: {e}")
    # finally:
    #     if os.path.exists("models_test_download"):
    #         shutil.rmtree("models_test_download")
    #         print("Cleaned up real model test directory.")

```
