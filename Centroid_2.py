import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
MARKER_SIZE = 0.1  # meters
REFERENCE_FILE = "pattern_reference.json"

# Camera calibration (replace with your real calibration values)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros(5)

# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# -----------------------------
# FUNCTIONS
# -----------------------------
def detect_markers(frame):
    """Detect markers, return dictionary with tvecs and centroids."""
    corners, ids, _ = detector.detectMarkers(frame)
    results = {}
    if ids is not None:
        ids = ids.flatten()
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
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
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Reference saved to {filename}")

def load_reference(filename=REFERENCE_FILE):
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as f:
        return json.load(f)

def compute_pattern_transform(current, reference):
    """Compute translation and rotation of pattern compared to reference."""
    if len(current) < 2:
        return None

    # Convert keys in reference to int for safety
    ref_vectors = {int(k): np.array(v) for k, v in reference["vectors"].items()}

    # Use the first detected marker as origin
    first_id = list(current.keys())[0]
    curr_origin = np.array(current[first_id]["tvec"])
    curr_vectors = {id_: np.array(data["tvec"]) - curr_origin for id_, data in current.items() if id_ in ref_vectors}

    # Translation (distance of origin to reference origin)
    translation = np.linalg.norm(curr_origin - np.array(reference["origin"]))

    # Rotation (average angle difference in XY plane)
    angles = []
    for id_, vec in curr_vectors.items():
        if id_ not in ref_vectors:
            continue
        ref_vec = ref_vectors[id_][:2]
        curr_vec = vec[:2]
        angle = np.arctan2(curr_vec[1], curr_vec[0]) - np.arctan2(ref_vec[1], ref_vec[0])
        angles.append(angle)
    rotation_deg = np.mean(angles) * 180 / np.pi if angles else 0.0

    return translation, rotation_deg

# -----------------------------
# MAIN WORKFLOW
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
            cv2.circle(frame, (cx, cy), 7, (0,0,255), -1)
            cv2.putText(frame, f"ID {id_}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            # Draw axes using correct OpenCV function
            rvec = np.array(data["rvec"]).reshape(3,1)
            tvec = np.array(data["tvec"]).reshape(3,1)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if len(detected) >= 2:
        origin = np.array(detected[1]["tvec"])
        vectors = {str(id_): (np.array(data["tvec"]) - origin).tolist() for id_, data in detected.items()}
        reference = {"origin": origin.tolist(), "vectors": vectors}
        save_reference(reference)
    else:
        print("Not enough markers detected for pattern reference.")

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
        if len(current) == 0:
            cv2.putText(frame, "No markers detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Check Pattern", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        transform = compute_pattern_transform(current, reference)
        if transform is not None:
            translation, rotation_deg = transform
            cv2.putText(frame, f"Translation: {translation:.3f} m", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Rotation: {rotation_deg:.2f} deg", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "Not enough markers for pattern", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        for id_, data in current.items():
            cx, cy = int(data["centroid"][0]), int(data["centroid"][1])
            cv2.circle(frame, (cx, cy), 7, (0,0,255), -1)
            cv2.putText(frame, f"ID {id_}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            rvec = np.array(data["rvec"]).reshape(3,1)
            tvec = np.array(data["tvec"]).reshape(3,1)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

        cv2.imshow("Check Pattern", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
