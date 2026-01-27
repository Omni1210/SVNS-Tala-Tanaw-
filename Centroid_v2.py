import cv2
import numpy as np
import json
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
MARKER_SIZE = 0.1  # meters
REFERENCE_FILE = "marker_reference.json"

# Camera calibration (replace with real calibration if available)
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

def save_reference(data, filename=REFERENCE_FILE):
    # Convert keys to strings for JSON
    str_data = {str(k): v for k, v in data.items()}
    with open(filename, "w") as f:
        json.dump(str_data, f, indent=2)
    print(f"Reference saved to {filename}")

def load_reference(filename=REFERENCE_FILE):
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as f:
        data = json.load(f)
    # Convert keys back to int
    return {int(k): v for k, v in data.items()}

def compare_with_reference(current, reference, pos_thresh=0.02, rot_thresh=5):
    """Compare current markers with reference. Returns dict of status per marker."""
    status = {}
    for id_, ref_data in reference.items():
        if id_ in current:
            # Translation delta
            t_current = np.array(current[id_]["tvec"])
            t_ref = np.array(ref_data["tvec"])
            delta_t = np.linalg.norm(t_current - t_ref)

            # Rotation delta (approximate)
            r_current = np.array(current[id_]["rvec"])
            r_ref = np.array(ref_data["rvec"])
            delta_r = np.linalg.norm(r_current - r_ref) * (180/np.pi)

            moved = delta_t > pos_thresh
            rotated = delta_r > rot_thresh

            status[id_] = {
                "moved": moved,
                "rotated": rotated,
                "delta_translation": delta_t,
                "delta_rotation": delta_r
            }
        else:
            status[id_] = {"moved": True, "rotated": True, "delta_translation": None, "delta_rotation": None}
    return status

# -----------------------------
# MAIN WORKFLOW
# -----------------------------

mode = input("Enter mode ('calibrate' or 'check'): ").strip().lower()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened. Check your camera index or driver.")
    exit()

# -----------------------------
# CALIBRATION MODE
# -----------------------------
if mode == "calibrate":
    print("Calibration mode: press ESC to capture reference")
    reference_data = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from camera")
            break

        detected = detect_markers(frame)

        # Draw markers
        for id_, data in detected.items():
            cx, cy = int(data["centroid"][0]), int(data["centroid"][1])
            cv2.circle(frame, (cx, cy), 7, (0,0,255), -1)
            cv2.putText(frame, f"ID {id_}", (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            # Draw axes
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              np.array(data["rvec"]), np.array(data["tvec"]), 0.05)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            if detected:
                reference_data = detected
                save_reference(reference_data)
            else:
                print("No markers detected. Reference not saved.")
            break

# -----------------------------
# CHECK MODE
# -----------------------------
elif mode == "check":
    reference = load_reference()
    if reference is None:
        print("No reference found. Please calibrate first.")
        exit()

    print("Checking mode: press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from camera")
            break

        current = detect_markers(frame)
        status = compare_with_reference(current, reference)

        for id_, s in status.items():
            if id_ in current:
                cx, cy = int(current[id_]["centroid"][0]), int(current[id_]["centroid"][1])
                color = (0,255,0) if not (s["moved"] or s["rotated"]) else (0,0,255)
                cv2.circle(frame, (cx, cy), 7, color, -1)
                text = f"ID {id_}"
                if s["moved"] or s["rotated"]:
                    text += " ✖"
                cv2.putText(frame, text, (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # Draw axes
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  np.array(current[id_]["rvec"]), np.array(current[id_]["tvec"]), 0.05)
            else:
                print(f"Marker {id_} missing!")

        cv2.imshow("Check Markers", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
