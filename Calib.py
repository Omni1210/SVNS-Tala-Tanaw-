import cv2
import numpy as np

KNOWN_DISTANCE = 0.3      # meters (measure with ruler)
KNOWN_MARKER_SIZE = 0.1  # meters

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        c = corners[0][0]
        pixel_width = np.linalg.norm(c[0] - c[1])

        focal_length = (pixel_width * KNOWN_DISTANCE) / KNOWN_MARKER_SIZE

        cv2.putText(frame,
                    f"Focal Length: {focal_length:.2f}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        print("Focal Length =", focal_length)

    cv2.imshow("Calibrate Focal Length", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

