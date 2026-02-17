import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import easyGraphing as eg



################
plot_offsetZ=eg.dataGrapher("z-offset","frame index", "z-offset (mm)")
plot_offsetX=eg.dataGrapher("X-offset","frame index", "x-offset (mm)")
plot_offsetY=eg.dataGrapher("y-offset","frame index", "y-offset (mm)")
################

numMetricDisplay=1
def displayMetric(metric, value, unit="m"):
    global numMetricDisplay
    cv2.putText(frame, metric+": " + str(value) + " "+unit, (50,numMetricDisplay*50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
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
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            # 5. Estimate 3D Pose using solvePnP
            _, rvec, tvec = cv2.solvePnP(obj_points, corners[i], mtx, dist, False, cv2.SOLVEPNP_IPPE_SQUARE)

            # Extract Position (Translation Vector)
            x, y, z = tvec.flatten()
            
            # Print data to console
            # print(f"ID: {ids[i][0]} | Pos (m): X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
            # print(tvec)
            # print("")
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
            ############################################################################################################
            # Estimate pose for each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, mtx, dist)

            # Get poses for the two specific markers (e.g., marker with id 1 and id 2)
            # Find indices
            idx1 = np.where(ids == 89)[0][0]
            idx2 = np.where(ids == 67)[0][0]

            rvec1, tvec1 = rvecs[idx1][0], tvecs[idx1][0]
            rvec2, tvec2 = rvecs[idx2][0], tvecs[idx2][0]
            R1, _ = cv2.Rodrigues(rvec1)
            R2, _ = cv2.Rodrigues(rvec2)
            T_cam_marker1 = np.eye(4)
            T_cam_marker1[0:3, 0:3] = R1
            T_cam_marker1[0:3, 3] = tvec1

            T_cam_marker2 = np.eye(4)
            T_cam_marker2[0:3, 0:3] = R2
            T_cam_marker2[0:3, 3] = tvec2
            # T_marker1_marker2 = T_marker1_cam @ T_cam_marker2
            # Where T_marker1_cam is the inverse of T_cam_marker1
            T_marker1_cam = np.linalg.inv(T_cam_marker1)
            T_marker1_marker2 = T_marker1_cam @ T_cam_marker2

            R_rel = T_marker1_marker2[0:3, 0:3]
            t_rel = T_marker1_marker2[0:3, 3]

            # Convert the relative rotation matrix back to a rotation vector if needed
            rvec_rel, _ = cv2.Rodrigues(R_rel)

            displayMetric("x-offset (red)",round(t_rel[0], 4)*1000,"mm")
            displayMetric("y-offset (green)",round(t_rel[1], 4)*1000,"mm")
            displayMetric("z-offset (blue)",round(t_rel[2], 4)*1000,"mm")
            plot_offsetZ.append(t_rel[2]*1000)
            plot_offsetX.append(t_rel[0]*1000)
            plot_offsetY.append(t_rel[1]*1000)

            # print(f"Relative Translation (Marker 2 relative to Marker 1):\n{t_rel}")
            # print(f"Relative Rotation Vector (Marker 2 relative to Marker 1):\n{rvec_rel}")

            ############################################################################################################


            ids_list = list(marker_positions.keys()) #each key is marker id and value is 3d position
            id1, id2 = ids_list[0], ids_list[1]

            p1 = marker_positions[id1] #3d position vector
            p2 = marker_positions[id2]

            distance = np.linalg.norm(p1 - p2) #linalg.norm computes euclidean length

            # print(f"Distance between marker {id1} and {id2}: {distance:.3f} m")

            # Draw a line between the two markers
            cv2.line(frame, marker_centers[id1], marker_centers[id2], (0, 255, 255), 4)


            displayMetric("distance",round(distance, 4)*1000,"mm") 

    # Show live video feed
    cv2.imshow("ArUco 3D Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

######################
# plot_offsetZ.graph()
# averageVal=np.mean(plot_offsetZ.dataList)
# print(averageVal)
# plt.axhline(y=averageVal, color='r', linestyle='--', linewidth=2)
# plt.show()
######################


#######
#use aruco 67 and 89
#it will crash if a diffent fiducial is detected


x_targetValue=500#these are in millimeters
y_targetValue=500
z_targetValue=500

plot_offsetZ.plot_VS_TargetValue(x_targetValue,0.5)
plot_offsetX.plot_VS_TargetValue(y_targetValue,0.5)
plot_offsetY.plot_VS_TargetValue(z_targetValue,0.5)

