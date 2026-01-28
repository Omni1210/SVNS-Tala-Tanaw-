import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
MARKER_SIZE = 0.07          # This works for you — only change if you re-measure one marker
TARGET_ALTITUDE = 0.5       # Target height for "LAND NOW" (50 cm)

REFERENCE_FILE = "marker_reference.json"
CALIB_FILE = "camera_calibration.npz"

# Video save folder
SAVE_FOLDER = r"D:\Users\Admin\Downloads\ERC\Sim_test\calib_sim"
os.makedirs(SAVE_FOLDER, exist_ok=True)

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

    # Translation vectors
    tvecs = [np.array(current[id_]["tvec"]) for id_ in common_ids]
    avg_tvec = np.mean(tvecs, axis=0)

    ref_tvecs = [np.array(reference[id_]["tvec"]) for id_ in common_ids]
    avg_ref_tvec = np.mean(ref_tvecs, axis=0)

    offsets = avg_tvec - avg_ref_tvec

    # Rotation error (proper math)
    rot_errors = []
    for id_ in common_ids:
        R1, _ = cv2.Rodrigues(np.array(current[id_]["rvec"]))
        R2, _ = cv2.Rodrigues(np.array(reference[id_]["rvec"]))
        R = R1 @ R2.T
        rvec_err, _ = cv2.Rodrigues(R)
        rot_errors.append(rvec_err.flatten())

    avg_rotation = np.mean(rot_errors, axis=0)

    # Distance
    distances = [np.linalg.norm(t) for t in tvecs]
    avg_distance = np.mean(distances)

    return offsets.flatten(), avg_rotation.flatten(), float(avg_distance), avg_tvec.flatten()


# -----------------------------
# MAIN
# -----------------------------
mode = input("Enter mode ('calibrate' or 'check'): ").strip().lower()

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
                save_reference(detected)
            else:
                print("No markers detected.")
        break

# CHECK MODE – Simple, accurate, no extras
if mode == "check":
    reference = load_reference()
    if reference is None:
        print("No reference found. Run 'calibrate' first.")
    else:
        print("Check mode active. Press ESC to exit.")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(SAVE_FOLDER, f"drone_guidance_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        print(f"Recording started → {video_filename}")

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
                    cv2.circle(frame, (cx, cy), 7, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID {id_}", (cx + 10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                      np.array(data["rvec"]), np.array(data["tvec"]), 0.05)

            offsets, rotation, avg_distance, avg_tvec = compute_offsets(current, reference)

            if offsets is not None:
                ox, oy, oz = offsets
                lateral_error = np.sqrt(ox**2 + oy**2)

                # Basic text feedback
                cv2.putText(frame, f"Center Offset: X={ox:.3f}m Y={oy:.3f}m Z={oz:.3f}m",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Avg Distance: {avg_distance:.3f} m",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Heading diff: {np.linalg.norm(rotation):.3f} rad",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                # Stray-away check
                if lateral_error > 0.10:
                    cv2.putText(frame, f"STRAYED AWAY: {lateral_error:.3f} m", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "CENTERED", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

                # Height / Descent Guidance (correct sign)
                descend_error = TARGET_ALTITUDE - oz  # positive = too far/high → descend
                if abs(descend_error) < 0.08:
                    cv2.putText(frame, "HEIGHT GOOD - LAND NOW", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                elif descend_error > 0:
                    cv2.putText(frame, f"DESCEND {descend_error:.3f} m", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
                else:
                    cv2.putText(frame, f"CLIMB {abs(descend_error):.3f} m", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                # Small upper-right prompt when ready to land
                if lateral_error < 0.05 and abs(descend_error) < 0.08 and abs(np.linalg.norm(rotation)) < 0.1:
                    text_x = frame.shape[1] - 220
                    text_y = 50
                    cv2.putText(frame, "LAND NOW!", (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                print(f"Height: {oz:.3f} m | Lateral: {lateral_error:.3f} m | Heading: {np.linalg.norm(rotation):.3f} rad")

            # Save frame to video
            out.write(frame)

            cv2.imshow("Landing Guidance (ESC to exit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        out.release()
        print(f"Video saved → {video_filename}")

cap.release()
cv2.destroyAllWindows()