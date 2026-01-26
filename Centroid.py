import cv2
import numpy as np

# -------------------------
# Marker & camera setup
# -------------------------
MARKER_SIZE = 0.1  # meters (100 mm)
FOCAL_LENGTH = 251  # in pixels
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Camera intrinsic matrix (no distortion)
cameraMatrix = np.array([[FOCAL_LENGTH, 0, FRAME_WIDTH/2],
                         [0, FOCAL_LENGTH, FRAME_HEIGHT/2],
                         [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros(5)

# ArUco dictionary & detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# -------------------------
# Iriun camera setup
# -------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# -------------------------
# Main loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame from camera")
        break

    # Detect markers
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None and len(ids) >= 1:  # at least 1 marker detected
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose for each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, cameraMatrix, distCoeffs
        )

        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.05)

        # Compute centroid of all detected markers
        centroid = np.mean(tvecs[:,0,:], axis=0)  # X, Y, Z in meters
        distance_to_centroid = np.linalg.norm(centroid)

        # Project centroid to 2D
        centroid_2d, _ = cv2.projectPoints(
            centroid.reshape((1,1,3)), np.zeros(3), np.zeros(3), cameraMatrix, distCoeffs
        )
        cx, cy = int(centroid_2d[0][0][0]), int(centroid_2d[0][0][1])

        # Draw centroid and distance
        cv2.circle(frame, (cx, cy), 7, (0,0,255), -1)
        cv2.putText(frame, f"Centroid Dist: {distance_to_centroid:.2f} m",
                    (cx-70, cy-20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

    cv2.imshow("Aruco Diamond Distance", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
