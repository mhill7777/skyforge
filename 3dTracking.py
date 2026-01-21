import cv2
import numpy as np
import matplotlib.pyplot as plt
#Here is a fancy comment that is testing out GIT
# 1. Load Calibration Data
with np.load('calibration_data2.npz') as data: #change directory if needed for testing
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

#Error + Velocity Tracking

TRUE_DISTANCE = 0.5  # meters (change to marker spacing)

measured_distances = []
distance_errors = []

velocities = []
velocity_errors = []
prev_pos = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect 2D Markers
    corners, ids, rejected = detector.detectMarkers(frame)

    marker_positions = {}   # stores tvecs by marker ID for distance between markers
    marker_centers = {}     # stores center pixel location for drawing a line


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
            
            marker_positions[ids[i][0]] = tvec.flatten()

            # Store marker center in pixels (for line drawing)
            c = corners[i][0]
            center_x = int(np.mean(c[:, 0]))
            center_y = int(np.mean(c[:, 1]))
            marker_centers[ids[i][0]] = (center_x, center_y)


    #code to get distance between 2 markers
    if len(marker_positions) >= 2:
        ids_list = list(marker_positions.keys()) #each key is marker id and value is 3d position
        id1, id2 = ids_list[0], ids_list[1]

        p1 = marker_positions[id1] #3d position vector
        p2 = marker_positions[id2]

        distance = np.linalg.norm(p1 - p2) #linalg.norm computes euclidean length

        print(f"Distance between marker {id1} and {id2}: {distance:.3f} m")

        # Draw a line between the two markers
        cv2.line(frame, marker_centers[id1], marker_centers[id2], (0, 255, 255), 4)

        # Error tracking
        error = abs(distance - TRUE_DISTANCE)
        measured_distances.append(distance)
        distance_errors.append(error) 

        #Velocity tracking
        fps = cap.get(cv2.CAP_PROP_FPS) #gets the frames per sec

        if fps != 0:    #calculates time between frames
            dt = 1 / fps
        else:  
            print("if didn't report fps right then just assume 30fps")
            dt = 1/30
        
        # choose marker 1 for velocity
        p = p1

        if prev_pos is not None:
            vel_vec = (p - prev_pos) / dt  #velocity vecotr (dif in position / time)
            vel_mag = np.linalg.norm(vel_vec) #convert velocity vector into a scalar speed (magnitude)
            velocities.append(vel_mag)

        prev_pos = p

        cv2.putText(frame, "Distance: " + str(round(distance, 4)) + " m", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show live video feed
    cv2.imshow("ArUco 3D Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

#all the error and velocity stats
print("Distance stats:")
print("Mean Distance:", np.mean(measured_distances))
print("Std Dev:", np.std(measured_distances))
print("Max Distance:", np.max(measured_distances))
print("Min Distance:", np.min(measured_distances))

print("Velocity stats:")
print("Mean Speed:", np.mean(velocities))
print("Std Dev:", np.std(velocities))
print("Max Speed:", np.max(velocities))
print("Min Speed:", np.min(velocities))


# plot cast
plt.plot(measured_distances)
plt.title("Measured Distance Over Time")
plt.xlabel("Frame Index")
plt.ylabel("Distance (m)")
plt.show()

plt.plot(velocities)
plt.title("Velocity Over Time")
plt.xlabel("Frame Index")
plt.ylabel("Speed (m/s)")
plt.show()
