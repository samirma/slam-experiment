import cv2

class MonocularCamera:
    """
    A class to interface with a monocular camera using OpenCV.
    It handles camera initialization, frame capture, and resource release.
    """
    def __init__(self, camera_id: int | str = 0):
        """
        Initializes the connection to the camera or video file.

        Args:
            camera_id (int or str): The ID of the camera (e.g., 0 for the default system camera)
                                    or the path to a video file. Defaults to 0.
        
        Raises:
            IOError: If the camera or video file cannot be opened.
        """
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera or video file with ID/path: {self.camera_id}")

    def get_frame(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """
        Reads and returns a single frame from the camera or video file.

        Returns:
            tuple[bool, numpy.ndarray | None]: 
                - success (bool): True if a frame was successfully read, False otherwise 
                                  (e.g., end of video file or camera disconnected).
                - frame (numpy.ndarray | None): The captured frame as a NumPy array (BGR format) 
                                                if successful, otherwise None.
        """
        success, frame = self.cap.read()
        return success, frame

    def release(self) -> None:
        """
        Releases the camera resource or video file.
        This method should be called explicitly when camera access is no longer needed,
        especially if not relying on the destructor (e.g., in long-running applications).
        """
        if self.cap.isOpened():
            self.cap.release()
            print(f"Camera/video {self.camera_id} released.")

    def __del__(self):
        """
        Destructor to ensure the camera resource is released when the object is garbage collected.
        It's good practice to call `release()` explicitly as well.
        """
        self.release()

if __name__ == '__main__':
    # Example usage:
    try:
        camera = MonocularCamera(0)
        print(f"Successfully opened camera with ID {camera.camera_id}")

        ret, frame = camera.get_frame()
        if ret:
            print("Successfully captured a frame.")
            # cv2.imshow("Test Frame", frame) # This would require a GUI environment
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print(f"Frame shape: {frame.shape}")
        else:
            print("Failed to capture a frame.")

        camera.release()
        print("Camera released.")

    except IOError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
