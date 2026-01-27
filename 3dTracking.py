import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os

#----------helper functions
#helps print on the camera feed
numMetricDisplay=1
def displayMetric(metric, value, unit="m"):
    global numMetricDisplay
    cv2.putText(frame, metric+": " + str(value) + " "+unit, (50,numMetricDisplay*50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    numMetricDisplay+=1
#helps make plots
def plot_VS_TargetValue(graphData, targetValue, title, ylabel, tolerance_m=0.0005):
    graphData_error = [x-targetValue for x in graphData]
    numOfMeasurements=range(len(graphData))
    upper_bound = tolerance_m
    lower_bound = -1 * tolerance_m
    fig, ax = plt.subplots()
    plt.plot(graphData_error)
    # 2. Plot the target value line
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Target Value')

    # 3. Plot the upper and lower tolerance lines (optional, can be done implicitly with fill_between)
    ax.axhline(upper_bound, color='gray', linestyle=':', linewidth=1, label='Tolerance Limit')
    ax.axhline(lower_bound, color='gray', linestyle=':', linewidth=1)

    # 4. Shade the tolerance band
    ax.fill_between(
        numOfMeasurements,
        lower_bound,
        upper_bound,
        color='red',
        alpha=0.2, # Adjust transparency
        label='Tolerance Range'
    )
    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel(ylabel)
    plt.show()
#---------------------

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

################# Definitions of plotting arrays (not all used)
measured_distances = []
distance_errors = []

velocities = []
velocity_errors = []
prev_pos = None

running_mse = []
frame_indices = []
####################################

data_per_id = {}
TRUE_DY = 0.0  #relative to each other
TRUE_DZ = 0.0  #relative to each other

while True:
    numMetricDisplay=1  #for printing information on their own rows
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

        for i in range(len(ids)): # loops for every detected aruco
            # 5. Estimate 3D Pose using solvePnP
            _, rvec, tvec = cv2.solvePnP(obj_points, corners[i], mtx, dist, False, cv2.SOLVEPNP_IPPE_SQUARE)
            
            marker_id = int(ids[i][0])
            x, y, z = tvec.flatten()
            
            ## printing rotation variables
            # rotationsVal=rvec.flatten()
            # displayMetric("id",ids[i])
            # displayMetric("r1",rotationsVal[0]*(180/math.pi),"")
            # displayMetric("r2",rotationsVal[1]*(180/math.pi),"")
            # displayMetric("r3",rotationsVal[2]*(180/math.pi),"")
            # print(rvec)

            #rotations in degrees
            r1, r2, r3 = (rvec.flatten() * (180 / math.pi))


            ## Extract Position (Translation Vector)
            # x, y, z = tvec.flatten()
            # displayMetric("x",x)
            # displayMetric("y",y)
            # displayMetric("z",z)
            
            # Print data to console
            #print(f"ID: {ids[i][0]} | Pos (m): X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
            # print(tvec)
            #print("")

             # Print rotation values to terminal
            print(f"ID {marker_id} | r1={r1:.3f}° r2={r2:.3f}° r3={r3:.3f}°")
            print(f"ID: {marker_id} | Pos (m): X={x:.3f}, Y={y:.3f}, Z={z:.3f}\n")

            # Store position
            marker_positions[marker_id] = np.array([x, y, z])

            # Store rotation
            if marker_id not in data_per_id:
                data_per_id[marker_id] = {
                    "x_err": [],
                    "y_err": [],
                    "z_err": [],
                    "r1": [],
                    "r2": [],
                    "r3": []
                }

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

        #xy_distance = np.sqrt(dx**2 + dy**2) #planar distance, ignore depth
        #z_distance = abs(dz) #how much farther from camera is one marker than other
        distance_3d = np.sqrt(dx**2 + dy**2 + dz**2) #physical distance between markers (euclidean)

        # Record errors per ID
        data_per_id[id1]["x_err"].append(abs(dx - TRUE_DISTANCE))
        data_per_id[id1]["y_err"].append(abs(dy - TRUE_DISTANCE))
        data_per_id[id1]["z_err"].append(abs(dz - TRUE_DISTANCE))

        data_per_id[id2]["x_err"].append(abs(dx - TRUE_DISTANCE))
        data_per_id[id2]["y_err"].append(abs(dy - TRUE_DISTANCE))
        data_per_id[id2]["z_err"].append(abs(dz - TRUE_DISTANCE))

        # displayMetric("Planar Dist", round(xy_distance, 4), "m")
        # displayMetric("Depth Dist", round(z_distance, 4), "m")
        # displayMetric("True Spacial Dist", round(distance_3d, 4), "m")


        # Draw a line between the two markers
        #cv2.line(frame, marker_centers[id1], marker_centers[id2], (0, 255, 255), 4)


        ################ updating plotting variables that depend on two fiducials
        # Error tracking
        error = abs(distance_3d - TRUE_DISTANCE)
        measured_distances.append(distance_3d)
        distance_errors.append(error)

        # RUNNING MS
        current_mse = np.mean(np.square(distance_errors))
        running_mse.append(current_mse)
        frame_indices.append(len(distance_errors))
        ############################################# 

        

        displayMetric("distance",round(distance_3d, 4))
    
    # Show live video feed
    cv2.imshow("ArUco 3D Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    
    df_mse_time = pd.DataFrame({
    "Frame": frame_indices,
    "Running_MSE": running_mse
})

cap.release()
cv2.destroyAllWindows()

# 8. Compute MSE per ID
# -----------------------------
results = []
for marker_id, data in data_per_id.items():
    mse_x = np.mean(np.square(data["x_err"]))
    mse_y = np.mean(np.square(data["y_err"]))
    mse_z = np.mean(np.square(data["z_err"]))

    results.append({
        "ID": marker_id,
        "MSE_X": mse_x,
        "MSE_Y": mse_y,
        "MSE_Z": mse_z
    })

# -----------------------------
# 9. Save Results to Text File
# -----------------------------
with open("results.txt", "w") as f:
    for marker_id, data in data_per_id.items():
        f.write(f"ID {marker_id} Errors:\n")
        f.write(f"  Mean X Error: {np.mean(data['x_err']):.4f} m\n")
        f.write(f"  Mean Y Error: {np.mean(data['y_err']):.4f} m\n")
        f.write(f"  Mean Z Error: {np.mean(data['z_err']):.4f} m\n")
        f.write("\n")
    print("saved to results.txt")

# -----------------------------
# 10. Save MSE to Excel
# -----------------------------
df = pd.DataFrame(results)

# If openpyxl isn't installed, run:
# pip install openpyxl
# with pd.ExcelWriter("mse_results.xlsx", engine="openpyxl") as writer:
#     df.to_excel(writer, sheet_name="Final_MSE_Per_ID", index=False)
#     df_mse_time.to_excel(writer, sheet_name="Running_MSE", index=False)

# print("Saved running MSE and final MSE to mse_results.xlsx")



#-------------------simple data plot template-----------------------
# plt.plot(arrayOfRecordedValues)
# plt.title("title")
# plt.xlabel("frame")
# plt.ylabel("dependent (unit)")
# plt.show()
#--------------------------------------------------------------------
plot_VS_TargetValue(measured_distances,TRUE_DISTANCE+MARKER_SIZE,"Measured Distance Over Time","Distance (m)")


# plt.plot(time_axis, velocities)
# plt.title("Velocity vs Time")
# plt.xlabel("Time (s)")
# plt.ylabel("Speed (m/s)")
# plt.show()


# with open("results.txt", "w") as f:
#     f.write("Distance Error Metrics\n")
#     f.write(f"Mean Error: {np.mean(distance_errors):.4f} m\n")
#     f.write(f"RMSE: {rmse_dist:.4f} m\n")
#     f.write(f"Std Dev: {np.std(distance_errors):.4f} m\n")
#     f.write(f"Max Error: {np.max(distance_errors):.4f} m\n")
#     f.write(f"Min Error: {np.min(distance_errors):.4f} m\n\n")

#     f.write("Velocity Metrics\n")
#     f.write(f"Mean Velocity: {np.mean(velocities):.4f} m/s\n")
#     f.write(f"Std Dev: {np.std(velocities):.4f} m/s\n")
#     f.write(f"Max Velocity: {np.max(velocities):.4f} m/s\n")
#     f.write(f"Min Velocity: {np.min(velocities):.4f} m/s\n")

#     if rmse_vel is not None:
#         f.write(f"Velocity RMSE: {rmse_vel:.4f} m/s\n")
#     else:
#         f.write("Velocity RMSE: N/A (no ground truth)\n")

#     f.write("\nTiming Info\n")
#     f.write(f"Estimated FPS: {fps}\n")
#     f.write(f"Delta Time (s): {dt:.4f}\n")
#     f.write(f"Total Velocity Samples: {len(velocities)}\n")

