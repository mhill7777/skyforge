import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


numMetricDisplay=1
def displayMetric(metric, value, unit="m"):
    global numMetricDisplay
    cv2.putText(frame, metric+": " + str(value) + " "+unit, (50,numMetricDisplay*50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    numMetricDisplay+=1

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

TRUE_DISTANCE = 0.0581  # meters (change to marker spacing)

measured_distances = []
distance_errors = []

velocities = []
velocity_errors = []
prev_pos = None

while True:
    numMetricDisplay=1#for printing information on their own rows
    ret, frame = cap.read()
    if not ret:
        break

    # Detect 2D Markers
    corners, ids, rejected = detector.detectMarkers(frame)

    marker_positions = {}   # stores tvecs by marker ID for distance between markers
    marker_centers = {}     # stores center pixel location for drawing a line


    if ids is not None:
        # Draw 2D green boxes and IDs
        cv2.aruco.drawDetectedMarkers(frame, corners)

        for i in range(len(ids)):
            # 5. Estimate 3D Pose using solvePnP
            _, rvec, tvec = cv2.solvePnP(obj_points, corners[i], mtx, dist, False, cv2.SOLVEPNP_IPPE_SQUARE)
            
            rotationsVal=rvec.flatten()
            #displayMetric("id",ids[i])
            #displayMetric("r1",rotationsVal[0]*(180/math.pi))
            #displayMetric("r2",rotationsVal[1]*(180/math.pi))
            #displayMetric("r3",rotationsVal[2]*(180/math.pi))
            print(rvec)

            # Extract Position (Translation Vector)
            x, y, z = tvec.flatten()
            
            # Print data to console
            print(f"ID: {ids[i][0]} | Pos (m): X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
            # print(tvec)
            print("")
            # 6. Draw 3D Axes on the Marker
            # 0.03 is the length of the axes lines in meters
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)
            
            marker_positions[ids[i][0]] = tvec.flatten()

            # Store marker center in pixels (for line drawing/all drawings)
            c = corners[i][0]
            center_x = int(np.mean(c[:, 0]))
            center_y = int(np.mean(c[:, 1]))
            marker_centers[ids[i][0]] = (center_x, center_y)

            # Compute pixel width/height
            width_px  = np.linalg.norm(c[0] - c[1])
            height_px = np.linalg.norm(c[1] - c[2])

            # Convert to meters
            z = tvec[2][0]
            fx = mtx[0, 0]
            fy = mtx[1, 1]

            width_m  = (width_px  * z) / fx
            height_m = (height_px * z) / fy

            cv2.putText(frame, f"ID: {ids[i][0]}", 
                        (center_x, center_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Z: {z:.2f} m", 
                        (center_x, center_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Size: {width_m:.3f}m", (center_x, center_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)




    #code to get distance between 2 markers
    if len(marker_positions) >= 2:
        ids_list = list(marker_positions.keys()) #each key is marker id and value is 3d position
        id1, id2 = ids_list[0], ids_list[1]

        p1 = marker_positions[id1] #3d position vector
        p2 = marker_positions[id2]

       # Component-wise distances (how far apart 2 markers against each axis)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]

        xy_distance = np.sqrt(dx**2 + dy**2) #planar distance, ignore depth
        z_distance = abs(dz) #how much farther from camera is one marker than other
        distance_3d = np.sqrt(dx**2 + dy**2 + dz**2) #physical distance between markers (euclidean)


        displayMetric("Planar Dist", round(xy_distance, 4), "m")
        displayMetric("Depth Dist", round(z_distance, 4), "m")
        displayMetric("True Spacial Dist", round(distance_3d, 4), "m")


        # Draw a line between the two markers
        #cv2.line(frame, marker_centers[id1], marker_centers[id2], (0, 255, 255), 4)

        # Error tracking
        error = abs(distance_3d - TRUE_DISTANCE)
        measured_distances.append(distance_3d)
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
            vel_vec = (p - prev_pos) / dt  #velocity vector (dif in position / time)
            vel_mag = np.linalg.norm(vel_vec) #convert velocity vector into a scalar speed (magnitude)
            velocities.append(vel_mag)

        prev_pos = p
        step = 50
        

        displayMetric("distance",round(distance_3d, 4))

    # Show live video feed
    cv2.imshow("ArUco 3D Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

# RMSE CALCULATIONS
rmse_dist = np.sqrt(np.mean(np.square(distance_errors))) #how far off distance estimates usually are(spacial accuracy)

# Only compute velocity RMSE if errors exist
if len(velocity_errors) > 0:
    rmse_vel = np.sqrt(np.mean(np.square(velocity_errors)))  #how far off velocity estimates usually are(motion accuracy)
else:
    rmse_vel = None

time_axis = np.arange(len(velocities)) * dt


#all the error and velocity stats
# print("Distance stats:")
# print("Mean Distance:", np.mean(measured_distances))
# print("Std Dev:", np.std(measured_distances))
# print("Max Distance:", np.max(measured_distances))
# print("Min Distance:", np.min(measured_distances))

# print("Velocity stats:")
# print("Mean Speed:", np.mean(velocities))
# print("Std Dev:", np.std(velocities))
# print("Max Speed:", np.max(velocities))

# plot cast
plt.plot(measured_distances)
plt.title("Measured Distance Over Time")
plt.xlabel("Frame Index")
plt.ylabel("Distance (m)")
plt.show()

plt.plot(time_axis, velocities)
plt.title("Velocity vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.show()


with open("results.txt", "w") as f:
    f.write("Distance Error Metrics\n")
    f.write(f"Mean Error: {np.mean(distance_errors):.4f} m\n")
    f.write(f"RMSE: {rmse_dist:.4f} m\n")
    f.write(f"Std Dev: {np.std(distance_errors):.4f} m\n")
    f.write(f"Max Error: {np.max(distance_errors):.4f} m\n")
    f.write(f"Min Error: {np.min(distance_errors):.4f} m\n\n")

    f.write("Velocity Metrics\n")
    f.write(f"Mean Velocity: {np.mean(velocities):.4f} m/s\n")
    f.write(f"Std Dev: {np.std(velocities):.4f} m/s\n")
    f.write(f"Max Velocity: {np.max(velocities):.4f} m/s\n")
    f.write(f"Min Velocity: {np.min(velocities):.4f} m/s\n")

    if rmse_vel is not None:
        f.write(f"Velocity RMSE: {rmse_vel:.4f} m/s\n")
    else:
        f.write("Velocity RMSE: N/A (no ground truth)\n")

    f.write("\nTiming Info\n")
    f.write(f"Estimated FPS: {fps}\n")
    f.write(f"Delta Time (s): {dt:.4f}\n")
    f.write(f"Total Velocity Samples: {len(velocities)}\n")

