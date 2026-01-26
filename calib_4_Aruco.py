import cv2
import numpy as np

# -------------------------
# Calibration setup
# -------------------------
KNOWN_DISTANCE = 0.4       # meters (distance from camera to diamond centroid)
MARKER_SIZE = 0.1          # meters (100 mm)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None and len(ids) >= 1:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Compute pixel widths and focal lengths for each marker
        focal_lengths = []
        all_corners = []
        for i in range(len(ids)):
            c = corners[i][0]
            pixel_width = np.linalg.norm(c[0] - c[1])
            f = (pixel_width * KNOWN_DISTANCE) / MARKER_SIZE
            focal_lengths.append(f)

            # Add all corners for centroid calculation
            all_corners.append(c)

        # Average focal length
        focal_length_avg = np.mean(focal_lengths)

        # Compute centroid from all corners
        all_corners_array = np.vstack(all_corners)  # shape (4*num_markers,2)
        centroid_pixel = np.mean(all_corners_array, axis=0)
        cx, cy = int(centroid_pixel[0]), int(centroid_pixel[1])

        # Draw centroid
        cv2.circle(frame, (cx, cy), 7, (0,0,255), -1)
        cv2.putText(frame,
                    f"Focal Length: {focal_length_avg:.2f}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)
        cv2.putText(frame,
                    f"Centroid Pixel: ({cx},{cy})",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,0,0),
                    2)

        print(f"Focal Length (avg): {focal_length_avg:.2f}, Centroid pixel: ({cx},{cy})")

    cv2.imshow("Calibrate Focal (4 markers centroid)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
