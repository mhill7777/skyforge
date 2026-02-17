import cv2
import numpy as np
import os

# --- 1. SETTINGS ---
fileName = "calibration_data"
# Use the exact dimensions you used to print your board
CHARUCO_BOARD = cv2.aruco.CharucoBoard(
    size=(9, 6),          # Number of squares (width, height)
    squareLength=0.02718,#0.02057,#0.03,    # Side length of squares (e.g., 0.03 for 30mm)
    markerLength=0.02032,#0.01495,#0.02,    # Side length of markers (e.g., 0.02 for 20mm)
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
)

# Initialize the detector
detector = cv2.aruco.CharucoDetector(CHARUCO_BOARD)

# --- 2. PROCESS IMAGES ---
all_charuco_corners = []
all_charuco_ids = []
image_size = None

image_folder = "../CalibrationImages/"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")]

for file in image_files:
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detectBoard handles both marker detection and corner interpolation
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    # We need at least 4 corners detected to be useful for calibration
    if charuco_ids is not None and len(charuco_ids) > 4:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)
        print(f"Accepted: {file}")
    else:
        print(f"Rejected: {file} (not enough corners)")

# --- 3. CALIBRATE ---
if len(all_charuco_corners) > 10:
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners, 
        charucoIds=all_charuco_ids, 
        board=CHARUCO_BOARD, 
        imageSize=image_size, 
        cameraMatrix=None, 
        distCoeffs=None
    )

    print(f"Calibration Complete. Reprojection Error: {ret:.4f}")
    
    # Save the parameters for your ArUco detection script
    np.savez(fileName+".npz", mtx=mtx, dist=dist)
    print("Files saved to 'calibration_data2.npz'")
else:
    print("Error: Not enough valid images found for calibration.")
