import cv2
import cv2.aruco as aruco
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
MARKER_SIZE = 0.1  # meters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Number of views to collect
NUM_VIEWS = 20

# Storage for calibration
obj_points = []  # 3D points in marker coordinate system
img_points = []  # 2D points in image

# Prepare object points for a single marker at origin
# Marker is on XY plane, Z=0
objp = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
], dtype=np.float32)

cap = cv2.VideoCapture(0)
collected = 0
print("Point your camera at the marker from different angles.")
print("Press SPACE to capture a view. ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners)

    cv2.putText(frame, f"Collected views: {collected}/{NUM_VIEWS}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imshow("Single Marker Calibration", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32 and ids is not None:  # SPACE
        # Use the first detected marker (only one expected)
        obj_points.append(objp)
        img_points.append(corners[0][0])
        collected += 1
        print(f"View {collected} captured.")
        if collected >= NUM_VIEWS:
            break

cap.release()
cv2.destroyAllWindows()

if collected < 5:
    print("Not enough views collected for calibration. Exiting.")
    exit()

# Camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, frame.shape[1::-1], None, None
)

print("\nCalibration complete!")
print("Camera matrix:")
print(camera_matrix)
print("Distortion coefficients:")
print(dist_coeffs.ravel())

# Optionally save to file
np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("Calibration saved to camera_calibration.npz")
