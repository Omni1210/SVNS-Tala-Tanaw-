import cv2
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R

# -----------------------------
# CONFIGURATION
# -----------------------------
MARKER_SIZE = 0.1  # meters
REFERENCE_FILE = "marker_reference.json"

# Camera calibration (replace with your real calibration values)
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
    """Detect markers, return ids, corners, centroids, and 3D poses."""
    corners, ids, _ = detector.detectMarkers(frame)
    results = {}
    if ids is not None:
        ids = ids.flatten()
        # Estimate 3D pose
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, camera_matrix, dist_coeffs
        )
        for i, id_ in enumerate(ids):
            centroid = np.mean(corners[i][0], axis=0)
            results[int(id_)] = {
                "centroid": centroid.tolist(),
                "rvec": rvecs[i][0].tolist(),
                "tvec": tvecs[i][0].tolist()
            }
    return results

def load_reference(filename=REFERENCE_FILE):
    if not os.path.exists(filename):
        print("Reference file not found!")
        return None
    with open(filename, "r") as f:
        return json.load(f)

def compute_offsets(current, reference):
    """Compute dx, dy, dz and rotation difference (yaw, pitch, roll)"""
    curr_pts = []
    ref_pts = []
    for id_, ref in reference.items():
        if int(id_) in current:
            curr_pts.append(current[int(id_)]["tvec"])
            ref_pts.append(ref["tvec"])
    if len(curr_pts) < 4:
        return None, None  # Not enough markers

    curr_pts = np.array(curr_pts)
    ref_pts = np.array(ref_pts)

    # Translation offset (pattern centroid)
    dx, dy, dz = np.mean(curr_pts, axis=0) - np.mean(ref_pts, axis=0)

    # Rotation offset using Kabsch
    H = (curr_pts - np.mean(curr_pts, axis=0)).T @ (ref_pts - np.mean(ref_pts, axis=0))
    U, S, Vt = np.linalg.svd(H)
    R_diff = Vt.T @ U.T

    # Correct for reflection
    if np.linalg.det(R_diff) < 0:
        Vt[-1,:] *= -1
        R_diff = Vt.T @ U.T

    rot = R.from_matrix(R_diff)
    yaw, pitch, roll = rot.as_euler('zyx', degrees=True)

    return (dx, dy, dz), (yaw, pitch, roll)


# -----------------------------
# MAIN WORKFLOW
# -----------------------------
reference = load_reference()
if reference is None:
    exit()

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("Checking mode: press ESC to exit")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current = detect_markers(frame)
    offsets, rotation = compute_offsets(current, reference)

    # Draw detected markers
    for id_, data in current.items():
        cx, cy = int(data["centroid"][0]), int(data["centroid"][1])
        cv2.circle(frame, (cx, cy), 7, (0,255,0), -1)
        cv2.putText(frame, f"ID {id_}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Display offsets if enough markers detected
    if offsets is not None:
        dx, dy, dz = offsets
        yaw, pitch, roll = rotation
        text = f"dx:{dx:.3f} dy:{dy:.3f} dz:{dz:.3f} | yaw:{yaw:.1f} pitch:{pitch:.1f} roll:{roll:.1f}"
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    else:
        cv2.putText(frame, "Not enough markers detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Drone Alignment", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
