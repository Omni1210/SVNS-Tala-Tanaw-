import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
MARKER_SIZE = 0.07          # ← CHANGE THIS! Measure ONE individual ArUco marker side (in meters)
# Example: if one marker is 42 mm wide → 0.042
# Measure from outer black edge to opposite outer black edge (including white quiet zone)

REFERENCE_FILE = "marker_reference.json"
CALIB_FILE = "camera_calibration.npz"

# Load calibration
if os.path.exists(CALIB_FILE):
    data = np.load(CALIB_FILE)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    print("Loaded calibration from", CALIB_FILE)
    print("RMS error was:", data.get("rms_error", data.get("rms", "unknown")))
else:
    print("Warning: Calibration file not found! Using fallback.")
    camera_matrix = np.array([
        [972.0740258,   0.,         329.55061338],
        [  0.,        971.46107308, 256.07779538],
        [  0.,          0.,           1.        ]
    ], dtype=np.float32)
    dist_coeffs = np.array([
        -0.690053623, 23.7744123, 0.00694494366, -0.0183891943, -166.352923
    ], dtype=np.float32)

# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# -----------------------------
# FUNCTIONS
# -----------------------------
def detect_markers(frame, undistort=False):
    if undistort:
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)
    
    corners, ids, rejected = detector.detectMarkers(frame)
    
    results = {}
    if ids is not None and len(ids) > 0:
        ids = ids.flatten()
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, camera_matrix, dist_coeffs
        )
        
        # Sub-pixel refinement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i in range(len(corners)):
            cv2.cornerSubPix(gray, corners[i].reshape(4, 1, 2), (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        for i, marker_id in enumerate(ids):
            centroid = np.mean(corners[i][0], axis=0)
            results[int(marker_id)] = {
                "centroid": centroid.tolist(),
                "rvec": rvecs[i][0].tolist(),
                "tvec": tvecs[i][0].tolist(),
                "corners": corners[i].tolist()
            }
    
    return results, corners if ids is not None else []

def save_reference(data, filename=REFERENCE_FILE):
    str_data = {str(k): v for k, v in data.items()}
    with open(filename, "w") as f:
        json.dump(str_data, f, indent=2)
    print(f"Reference saved to {filename}")

def load_reference(filename=REFERENCE_FILE):
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

def compute_offsets(current, reference):
    common_ids = [id_ for id_ in current if id_ in reference]
    if not common_ids:
        return None, None, None, None

    tvecs = [np.array(current[id_]["tvec"]) for id_ in common_ids]
    avg_tvec = np.mean(tvecs, axis=0)

    ref_tvecs = [np.array(reference[id_]["tvec"]) for id_ in common_ids]
    avg_ref_tvec = np.mean(ref_tvecs, axis=0)

    offsets = avg_tvec - avg_ref_tvec

    rot_vectors = [np.array(current[id_]["rvec"]) - np.array(reference[id_]["rvec"]) for id_ in common_ids]
    avg_rotation = np.mean(rot_vectors, axis=0)

    distances = [np.linalg.norm(t) for t in tvecs]
    avg_distance = np.mean(distances)

    return offsets.flatten(), avg_rotation.flatten(), float(avg_distance), avg_tvec.flatten()

# -----------------------------
# MAIN
# -----------------------------
mode = input("Enter mode ('calibrate' or 'check'): ").strip().lower()

# Camera open with fallback
cap = None
for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
    for idx in range(4):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            print(f"Opened camera index {idx} with backend {backend}")
            break
    if cap is not None and cap.isOpened():
        break

if cap is None or not cap.isOpened():
    print("Failed to open camera.")
    exit()

print("Press ESC to capture reference (calibrate) or exit (check)")

reference_data = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        break

    detected, marker_corners = detect_markers(frame, undistort=False)

    if len(marker_corners) > 0:
        aruco.drawDetectedMarkers(frame, marker_corners,
                                  np.array(list(detected.keys()), dtype=np.int32),
                                  borderColor=(0, 255, 0))

        for id_, data in detected.items():
            cx, cy = int(data["centroid"][0]), int(data["centroid"][1])
            cv2.circle(frame, (cx, cy), 7, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {id_}", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              np.array(data["rvec"]), np.array(data["tvec"]), 0.05)

    cv2.putText(frame, f"Detected: {len(detected)} markers", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("ArUco Tracker (ESC to capture/exit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        if mode == "calibrate":
            if detected:
                reference_data = detected
                save_reference(reference_data)
            else:
                print("No markers detected.")
        break

# CHECK MODE
if mode == "check":
    reference = load_reference()
    if reference is None:
        print("No reference found. Run 'calibrate' first.")
    else:
        print("Check mode active. Press ESC to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current, marker_corners = detect_markers(frame, undistort=False)

            if len(marker_corners) > 0:
                aruco.drawDetectedMarkers(frame, marker_corners,
                                          np.array(list(current.keys()), dtype=np.int32),
                                          borderColor=(0, 255, 0))

                for id_, data in current.items():
                    cx, cy = int(data["centroid"][0]), int(data["centroid"][1])
                    color = (0, 255, 0) if id_ in reference else (0, 0, 255)
                    cv2.circle(frame, (cx, cy), 7, color, -1)
                    cv2.putText(frame, f"ID {id_}", (cx + 10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                      np.array(data["rvec"]), np.array(data["tvec"]), 0.05)

            offsets, rotation, avg_distance, avg_tvec = compute_offsets(current, reference)

            if offsets is not None:
                ox, oy, oz = offsets
                cv2.putText(frame, f"Center Offset: X={ox:.3f}m Y={oy:.3f}m Z={oz:.3f}m",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Avg Distance: {avg_distance:.3f} m",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Heading diff: {np.linalg.norm(rotation):.3f} rad",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                # Debug: individual distances
                print("Individual distances:")
                for id_ in current:
                    dist = np.linalg.norm(np.array(current[id_]["tvec"]))
                    print(f"  ID {id_}: {dist:.3f} m")

            cv2.imshow("Check Markers (ESC to exit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()