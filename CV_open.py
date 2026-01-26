import cv2
import numpy as np

KNOWN_MARKER_SIZE = 0.1  # meters
FOCAL_LENGTH = 855  # approximate

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
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

        distance = (KNOWN_MARKER_SIZE * FOCAL_LENGTH) / pixel_width

        cv2.putText(frame,
                    f"Distance: {distance:.2f} m",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

    cv2.imshow("Approx Distance", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
