import cv2
import numpy as np
import os

# ================= CONFIG =================
CHESSBOARD = (9, 6)                  # Matches pattern.png
SQUARE_SIZE = 0.028                  # ← CHANGE TO YOUR measured square size in meters!
CAM_INDEX = 1                        # ← CHANGE TO your working index from tester!
MIN_IMAGES = 8
SAVE_FOLDER = r"D:\Users\Admin\Downloads\ERC\Calib_img"
os.makedirs(SAVE_FOLDER, exist_ok=True)
# ==========================================

objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = []
imgpoints = []

# Try different backends if needed: cv2.CAP_DSHOW (default Windows), cv2.CAP_MSMF
for cam_index in range(3):
    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        print(f"Using camera index {cam_index}")
        break
else:
    print("Camera could not be opened. Check camera index or if it's in use.")
    exit()


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print("\n=== LIVE CALIBRATION START ===")
print("Instructions:")
print("1. Point iPhone at full-screen chessboard.")
print("2. When GREEN corners appear over ALL squares → press 's' to save frame.")
print("3. Collect 10–20+ varied poses:")
print("   - Closer / farther")
print("   - Tilted left/right/up/down")
print("   - Pattern near edges/corners of view")
print("   - Slight rotations")
print("4. Press 'q' to stop and calibrate.")
print("=============================\n")

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_c, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)

    display = frame.copy()
    if ret_c:
        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(display, CHESSBOARD, corners_refined, ret_c)
        cv2.putText(display, f"Saved so far: {len(objpoints)}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('Live - s=save good frame, q=quit & calibrate', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and ret_c:
        fname = os.path.join(SAVE_FOLDER, f"good_{frame_num:03d}.jpg")
        cv2.imwrite(fname, frame)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        print(f"SAVED: {fname}  (total saved: {len(objpoints)})")
        frame_num += 1
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nCollected {len(objpoints)} good frames.")

if len(objpoints) < MIN_IMAGES:
    print(f"Not enough! Need at least {MIN_IMAGES}. Run again for more variety.")
else:
    print("Starting calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print(f"RMS reprojection error: {ret:.4f} pixels")
    print("  → Great if <0.8, good if <1.2, retry if >1.5")
    print("\nCamera Matrix:\n", np.round(mtx, 4))
    print("\nDistortion Coeffs:\n", np.round(dist, 6))

    np.savez(r"D:\Users\Admin\Downloads\ERC\camera_calibration.npz",
             camera_matrix=mtx, dist_coeffs=dist,
             rms=ret, image_size=(gray.shape[1], gray.shape[0]))

    # Quick check on last saved image
    if frame_num > 0:
        last_file = os.path.join(SAVE_FOLDER, f"good_{frame_num-1:03d}.jpg")
        test_img = cv2.imread(last_file)
        if test_img is not None:
            undist = cv2.undistort(test_img, mtx, dist, None, mtx)
            cv2.imshow('Last Saved Original', test_img)
            cv2.imshow('Undistorted (check straighter edges)', undist)
            cv2.waitKey(0)