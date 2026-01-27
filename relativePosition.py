import cv2
import numpy as np

# Load camera calibration data (mtx, dist)
with np.load('camera_calibration.npz') as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

# Define marker parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
marker_length = 0.05 # e.g., 5 cm

# Detect markers in the image/frame
corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

if ids is not None and len(ids) >= 2:
    # Estimate pose for each marker
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

    # Get poses for the two specific markers (e.g., marker with id 1 and id 2)
    # Find indices
    idx1 = np.where(ids == 1)[0][0]
    idx2 = np.where(ids == 2)[0][0]

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

    print(f"Relative Translation (Marker 2 relative to Marker 1):\n{t_rel}")
    print(f"Relative Rotation Vector (Marker 2 relative to Marker 1):\n{rvec_rel}")







