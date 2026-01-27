import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
MARKER_SIZE = 0.042          # Your measured value (one individual ArUco marker in meters)

REFERENCE_FILE = "marker_reference.json"
CALIB_FILE = "camera_calibration.npz"

# Video save folder (your path)
VIDEO_SAVE_FOLDER = r"D:\Users\Admin\Downloads\ERC\Sim_test\calib_sim"

# Make sure the folder exists
os.makedirs(VIDEO_SAVE_FOLDER, exist_ok=True)

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

def get_cardinal_directions(current):
    if len(current) != 4:
        return {}

    cents = {mid: np.array(d["centroid"]) for mid, d in current.items()}
    by_y = sorted(cents.items(), key=lambda x: x[1][1])
    by_x = sorted(cents.items(), key=lambda x: x[1][0])

    return {
        'North': by_y[0][0],
        'South': by_y[-1][0],
        'West':  by_x[0][0],
        'East':  by_x[-1][0],
    }

def compute_heading_error(current, reference, directions):
    if 'North' not in directions:
        return None

    north_id = directions['North']
    if north_id not in current or north_id not in reference:
        return None

    curr_rvec = np.array(current[north_id]["rvec"])
    ref_rvec  = np.array(reference[north_id]["rvec"])

    R_curr, _ = cv2.Rodrigues(curr_rvec)
    R_ref,  _ = cv2.Rodrigues(ref_rvec)

    R_rel = R_curr @ R_ref.T

    yaw_rad = np.arctan2(R_rel[1, 0], R_rel[0, 0])
    heading_error_deg = np.degrees(yaw_rad)
    heading_error_deg = (heading_error_deg + 180) % 360 - 180

    return heading_error_deg

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

# CHECK MODE – Landing Guidance + Video Recording
if mode == "check":
    reference = load_reference()
    if reference is None:
        print("No reference found. Run 'calibrate' first.")
    else:
        print("Check mode active – Landing guidance ON. Press ESC to exit.")
        
        # Start video recording in your specified folder
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(VIDEO_SAVE_FOLDER, f"drone_guidance_{timestamp}.mp4")
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
                    color = (0, 255, 0) if id_ in reference else (0, 0, 255)
                    cv2.circle(frame, (cx, cy), 7, color, -1)
                    cv2.putText(frame, f"ID {id_}", (cx + 10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                      np.array(data["rvec"]), np.array(data["tvec"]), 0.05)

            offsets, rotation, avg_distance, avg_tvec = compute_offsets(current, reference)
            directions = get_cardinal_directions(current)

            # Directions
            if directions:
                print("\nDirections:")
                for dir_name, mid in directions.items():
                    print(f"  {dir_name}: ID {mid}")
                for dir_name, mid in directions.items():
                    if mid in current:
                        cx, cy = map(int, current[mid]["centroid"])
                        cv2.putText(frame, dir_name.upper(), (cx + 35, cy - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2, cv2.LINE_AA)

            # Landing Guidance: Position & Rotation
            if offsets is not None:
                ox, oy, oz = offsets
                lateral_error = np.sqrt(ox**2 + oy**2)

                if lateral_error < 0.05 and abs(oz) < 0.10:
                    status = "GOOD TO LAND"
                    color = (0, 255, 0)
                elif lateral_error < 0.15 and abs(oz) < 0.30:
                    status = "ADJUST SLIGHTLY"
                    color = (0, 255, 255)
                else:
                    status = "ALIGN / TOO FAR"
                    color = (0, 0, 255)

                cv2.putText(frame, f"STATUS: {status}", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                cv2.putText(frame, f"Lateral: {lateral_error:.3f} m", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Height: {oz:.3f} m", (10, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Correction arrow
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                arrow_end = (int(center[0] + ox * 1200), int(center[1] + oy * 1200))
                cv2.arrowedLine(frame, center, arrow_end, color, 6, tipLength=0.4)

            heading_error = compute_heading_error(current, reference, directions)
            if heading_error is not None:
                if abs(heading_error) < 5:
                    rot_text = "HEADING ALIGNED"
                    rot_color = (0, 255, 0)
                else:
                    dir_text = "RIGHT" if heading_error > 0 else "LEFT"
                    rot_text = f"Rotate {dir_text} {abs(heading_error):.1f}°"
                    rot_color = (255, 215, 0)

                cv2.putText(frame, rot_text, (10, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, rot_color, 2)

                # Rotation indicator
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                radius = 80
                start_angle = 90
                end_angle = 90 + heading_error * 3
                cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle,
                            rot_color, 4, cv2.LINE_AA)

            # Debug distances
            print("Individual distances:")
            for id_ in current:
                dist = np.linalg.norm(np.array(current[id_]["tvec"]))
                print(f"  ID {id_}: {dist:.3f} m")

            # Save frame to video
            out.write(frame)

            cv2.imshow("Landing Guidance (ESC to exit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Finish video
        out.release()
        print(f"Video saved → {video_filename}")

cap.release()
cv2.destroyAllWindows()