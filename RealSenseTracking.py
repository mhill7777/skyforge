import pyrealsense2 as rs
import cv2
import numpy as np
import math
import time
import pandas as pd
import matplotlib.pyplot as plt

#test with only one aurco marker!

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

MARKER_SIZE = 0.05057  # meters

obj_points = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
], dtype=np.float32)

# RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# Get camera intrinsics
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy

mtx = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])

dist = np.zeros(5)  # RealSense RGB is rectified

print("RealSense intrinsics loaded")

# ---- DATA STORAGE ----
results = []
frame_count = 0
error_history = []

# Main loop
try:
    while True:
        frame_count += 1

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners)

            for i in range(len(ids)):
                _, rvec, tvec = cv2.solvePnP(
                    obj_points, corners[i], mtx, dist,
                    False, cv2.SOLVEPNP_IPPE_SQUARE
                )

                z_pnp = tvec[2][0]

                c = corners[i][0]
                cx_px = int(np.mean(c[:, 0]))
                cy_px = int(np.mean(c[:, 1]))

                z_rs = depth_frame.get_distance(cx_px, cy_px)
                
                error = abs(z_pnp - z_rs)
                error_history.append(error)
                running_mse = np.mean(np.square(error_history))
                # marker size in pixels
                width_px = int(abs(c[0][0] - c[1][0]))
                height_px = int(abs(c[0][1] - c[3][1]))

                # ---- STORE DATA ----
                results.append({
                    "timestamp": time.time(),
                    "frame": frame_count,
                    "marker_id": int(ids[i][0]),
                    "z_pnp": float(z_pnp),
                    "z_rs": float(z_rs),
                    "error": float(abs(z_pnp - z_rs)),
                    "width_px": width_px,
                    "height_px": height_px,
                    "rvec_x": float(rvec[0][0]),
                    "rvec_y": float(rvec[1][0]),
                    "rvec_z": float(rvec[2][0])
                })

                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)

                cv2.putText(frame, f"ID {ids[i][0]}",
                            (cx_px, cy_px - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.putText(frame, f"PnP Z: {z_pnp:.3f} m",
                            (cx_px, cy_px - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                cv2.putText(frame, f"RS Z: {z_rs:.3f} m",
                            (cx_px, cy_px + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                cv2.putText(frame, f"Error: {abs(z_pnp - z_rs):.3f} m",
                            (cx_px, cy_px + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("RealSense Depth vs ArUco PnP", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# ---- WRITE TO EXCEL ----
df = pd.DataFrame(results)
df.to_excel("depth_comparison.xlsx", index=False)

print("Saved Excel: depth_comparison.xlsx")

# ---- PLOT RESULTS ----
plt.figure()
plt.plot(df["frame"], df["z_pnp"], label="PnP Depth")
plt.plot(df["frame"], df["z_rs"], label="RealSense Depth")
plt.title("PnP vs RealSense Depth")
plt.xlabel("Frame")
plt.ylabel("Depth (m)")
plt.legend()
plt.show()

plt.figure()
plt.plot(df["frame"], df["error"])
plt.title("Depth Error Over Time")
plt.xlabel("Frame")
plt.ylabel("Error (m)")
plt.show()
