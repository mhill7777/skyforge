import cv2
import numpy as np
#Here is a fancy comment that is testing out GIT
# 1. Load Calibration Data
with np.load('calibration_data.npz') as data: #change directory if needed for testing
    mtx = data['mtx']
    dist = data['dist']

# 2. Setup ArUco Detector (Standard for 2026)
# Match this to the dictionary you used to print your markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# 3. Define Physical Marker Geometry
MARKER_SIZE = 0.05057  # Actual side length in meters (e.g., 5cm)
# Object points for a square marker centered at (0,0,0)
obj_points = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
], dtype=np.float32)

# 4. Start Video Feed
cap = cv2.VideoCapture(0) # Change 0 to your camera index if needed

print("Starting detection. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect 2D Markers
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        # Draw 2D green boxes and IDs
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            # 5. Estimate 3D Pose using solvePnP
            _, rvec, tvec = cv2.solvePnP(obj_points, corners[i], mtx, dist, False, cv2.SOLVEPNP_IPPE_SQUARE)

            # Extract Position (Translation Vector)
            x, y, z = tvec.flatten()
            
            # Print data to console
            print(f"ID: {ids[i][0]} | Pos (m): X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
            # print(tvec)
            print("")
            # 6. Draw 3D Axes on the Marker
            # 0.03 is the length of the axes lines in meters
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)
            

            # Optional: Display Z-distance on the video frame
            cv2.putText(frame, f"Z: {z:.2f}m", (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)

    # Show live video feed
    cv2.imshow("ArUco 3D Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
