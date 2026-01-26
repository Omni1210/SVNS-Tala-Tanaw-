import cv2
import numpy as np
import json
import time
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
MARKER_SIZE = 0.1  # meters
REFERENCE_FILE = "aruco_pattern_reference.json"
CAMERA_ID = 1      # change if using Iriun webcam
FPS_RECORD = 2     # frames per second for reference
COUNTDOWN = 10     # seconds before recording starts

# Camera calibration (replace with your real calibration)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros(5)  # assume no lens distortion

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# -----------------------------
# FUNCTIONS
# -----------------------------
def detect_markers(frame):
    """Detect markers, return dict of ids with rvec and tvec."""
    corners, ids, _ = detector.detectMarkers(frame)
    results = {}
    if ids is not None:
        ids = ids.flatten()
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, camera_matrix, dist_coeffs
        )
        for i, id_ in enumerate(ids):
            results[int(id_)] = {
                "rvec": rvecs[i][0].tolist(),
                "tvec": tvecs[i][0].tolist()
            }
    return results

def save_reference(data, filename=REFERENCE_FILE):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Reference saved to {filename}")

# -----------------------------
# MAIN WORKFLOW
# -----------------------------
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

# Countdown before recording
print(f"Starting in {COUNTDOWN} seconds. Prepare your pattern...")
for i in range(COUNTDOWN, 0, -1):
    print(i)
    time.sleep(1)

reference_frames = []
last_record = 0

print("Recording reference. Press ESC to finish.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    markers = detect_markers(frame)

    # Draw markers
    for id_, data in markers.items():
        cx, cy = int(data["tvec"][0]*500 + 320), int(data["tvec"][1]*500 + 240)  # rough visualization
        cv2.putText(frame, f"ID {id_}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Reference Calibration", frame)

    # Record at FPS_RECORD
    if time.time() - last_record > 1.0 / FPS_RECORD and markers:
        reference_frames.append(markers)
        last_record = time.time()

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# Save all frames
save_reference(reference_frames)
