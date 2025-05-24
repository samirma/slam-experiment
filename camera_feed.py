import cv2

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop to continuously read frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame in a window titled "Camera Feed"
    cv2.imshow("Camera Feed", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
