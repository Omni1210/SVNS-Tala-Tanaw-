import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R

# -----------------------------
# CONFIG
# -----------------------------
MARKER_SIZE = 0.1  # meters
REFERENCE_FILE = "marker_reference.json"

# Camera calibration (replace with real calibration values)
# Load real calibration
data = np.load("camera_calibration.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# -----------------------------
# FUNCTIONS
# -----------------------------

def detect_markers(frame):
    """Detect markers and return a dict {id: {'centroid':[x,y], 'tvec':[x,y,z], 'rvec':[x,y,z]}}"""
    corners, ids, _ = detector.detectMarkers(frame)
    results = {}
    if ids is not None:
        ids = ids.flatten()
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
        for i, id_ in enumerate(ids):
            centroid = np.mean(corners[i][0], axis=0)
            results[int(id_)] = {
                "centroid": centroid.tolist(),
                "rvec": rvecs[i][0].tolist(),
                "tvec": tvecs[i][0].tolist()
            }
    return results

def save_reference(data, filename=REFERENCE_FILE):
    # Save with string keys for JSON
    str_data = {str(k): v for k, v in data.items()}
    with open(filename, "w") as f:
        json.dump(str_data, f, indent=2)
    print(f"Reference saved to {filename}")

def load_reference(filename=REFERENCE_FILE):
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as f:
        data = json.load(f)
    # convert keys to int
    return {int(k): v for k, v in data.items()}

def compute_offsets(current, reference):
    """Compute translation offset, rotation, and average distance to camera"""
    common_ids = [id_ for id_ in current if id_ in reference]
    if not common_ids:
        return None, None, None

    curr_pts = np.array([current[id_]["tvec"] for id_ in common_ids])
    ref_pts = np.array([reference[id_]["tvec"] for id_ in common_ids])

    # Translation offset
    offsets = np.mean(curr_pts - ref_pts, axis=0)

    # Rotation estimation (approximate)
    R_curr = np.eye(3)
    R_ref = np.eye(3)
    # Using single marker rotation difference (average)
    rot_vectors = [np.array(current[id_]["rvec"]) - np.array(reference[id_]["rvec"]) for id_ in common_ids]
    rotation = np.mean(rot_vectors, axis=0)

    # Distance from camera to marker centroids
    distances = [np.linalg.norm(np.array(current[id_]["tvec"])) for id_ in common_ids]
    distance = np.mean(distances)

    return offsets, rotation, distance

# -----------------------------
# MAIN
# -----------------------------
mode = input("Enter mode ('calibrate' or 'check'): ").strip().lower()
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if mode == "calibrate":
    print("Calibration mode: press ESC to capture reference")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected = detect_markers(frame)
        for id_, data in detected.items():
            cx, cy = int(data["centroid"][0]), int(data["centroid"][1])
            color = (0, 255, 0)
            cv2.circle(frame, (cx, cy), 7, color, -1)
            cv2.putText(frame, f"ID {id_}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Draw approximate axes
            cv2.line(frame, (cx, cy), (cx + 30, cy), (0,0,255), 2)
            cv2.line(frame, (cx, cy), (cx, cy + 30), (0,255,0), 2)

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if detected:
        save_reference(detected)
    else:
        print("No markers detected. Reference not saved.")

elif mode == "check":
    reference = load_reference()
    if reference is None:
        print("No reference found. Please calibrate first.")
        exit()

    print("Checking mode: press ESC to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current = detect_markers(frame)

        offsets, rotation, distance = compute_offsets(current, reference)

        for id_, data in current.items():
            cx, cy = int(data["centroid"][0]), int(data["centroid"][1])
            color = (0, 255, 0)
            if id_ not in reference:
                color = (0, 0, 255)
            cv2.circle(frame, (cx, cy), 7, color, -1)
            cv2.putText(frame, f"ID {id_}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display offsets and distance
        if offsets is not None:
            text = f"Offset: X={offsets[0]:.3f}, Y={offsets[1]:.3f}, Z={offsets[2]:.3f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            text2 = f"Rotation vector: {rotation}"
            cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            text3 = f"Distance: {distance:.3f} m"
            cv2.putText(frame, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow("Check Markers", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
