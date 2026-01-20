import cv2 as cv

# Create a VideoCapture object for the default camera (index 0)
cap = cv.VideoCapture(0)

# Check if the camera opened successfully 
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Loop to capture and display frames
while True:
    ret, frame = cap.read() # Capture frame-by-frame

    if not ret:
        print("Can't receive frame. Exiting ...") 
        break 

    # Convert the image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    parameters = cv.aruco.DetectorParameters()

    # Create the ArUco detector
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Print the detected markers
    print("Detected markers:", ids)
    if ids is not None:
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
    if len(rejected) > 0:
        cv.aruco.drawDetectedMarkers(frame, rejected, borderColor=(100, 0, 240))

    cv.imshow('Live Camera Feed', frame) # Display the frame

    if cv.waitKey(1) == ord('q'): # Exit loop on 'q' press
        break

# Release the camera and close all windows
cap.release()
cv.destroyAllWindows()
