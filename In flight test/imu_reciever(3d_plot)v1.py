import requests
import json
import time
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from collections import deque

# =============================
# CONFIGURATION
# =============================
PHYPH0X_IP = "192.168.100.229"      # Your Phyphox IP
PHYPH0X_PORT = 8080
BASE_URL = f"http://{PHYPH0X_IP}:{PHYPH0X_PORT}/get?"

SENSORS = [
    "accX", "accY", "accZ",           # Generic Acceleration
    "linAccX", "linAccY", "linAccZ",  # Linear Acceleration (used for Kalman)
    "magX", "magY", "magZ"            # Magnetometer
]

POLL_INTERVAL = 0.1               # seconds
MAX_TRAIL = 50                    # trail length
CUBE_SIZE = 50.0                  # ±50 m = 100 × 100 × 100 m cube
LOG_TO_CSV = True
CSV_FILE = "phyphox_imu_log.csv"

# =============================
# KALMAN FILTER (per axis)
# =============================
class Kalman1D:
    def __init__(self, process_noise=0.01, measurement_noise=0.1, initial_estimate=0.0):
        self.x = initial_estimate  # state (position or velocity)
        self.P = 1.0               # error covariance
        self.Q = process_noise     # process noise
        self.R = measurement_noise # measurement noise

    def update(self, measurement):
        # Predict
        self.P = self.P + self.Q

        # Update
        K = self.P / (self.P + self.R)  # Kalman gain
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P

        return self.x

# Kalman filters for position (one per axis)
kalman_pos_x = Kalman1D(process_noise=0.05, measurement_noise=0.5)
kalman_pos_y = Kalman1D(process_noise=0.05, measurement_noise=0.5)
kalman_pos_z = Kalman1D(process_noise=0.05, measurement_noise=0.5)

# =============================
# POSITION & VELOCITY
# =============================
position = np.array([0.0, 0.0, 0.0])
velocity = np.array([0.0, 0.0, 0.0])
dt = POLL_INTERVAL

position_queue = deque(maxlen=MAX_TRAIL)

# =============================
# CSV SETUP
# =============================
if LOG_TO_CSV:
    with open(CSV_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp"] + SENSORS + ["posX", "posY", "posZ"])

# =============================
# PHYPH0X POLLING FUNCTION
# =============================
def fetch_phyphox_data():
    global position, velocity

    try:
        url = BASE_URL + "&".join(SENSORS)
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        data = json.loads(response.text)

        buffer = data.get("buffer", {})
        if not buffer:
            print("Buffer empty — recording active?")
            return False

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Linear acceleration
        lin_ax = buffer.get("linAccX", {}).get("buffer", [0.0])[-1]
        lin_ay = buffer.get("linAccY", {}).get("buffer", [0.0])[-1]
        lin_az = buffer.get("linAccZ", {}).get("buffer", [0.0])[-1]

        lin_acc = np.array([lin_ax, lin_ay, lin_az])

        # Velocity update
        velocity += lin_acc * dt

        # Position update with Kalman smoothing
        position[0] = kalman_pos_x.update(position[0] + velocity[0] * dt)
        position[1] = kalman_pos_y.update(position[1] + velocity[1] * dt)
        position[2] = kalman_pos_z.update(position[2] + velocity[2] * dt)

        position_queue.append(position.copy())

        # Magnetometer
        mag_x = buffer.get("magX", {}).get("buffer", [0.0])[-1]
        mag_y = buffer.get("magY", {}).get("buffer", [0.0])[-1]
        mag_z = buffer.get("magZ", {}).get("buffer", [0.0])[-1]

        # Print
        print(f"[{ts}] "
              f"LinAcc: {lin_ax:+6.3f} {lin_ay:+6.3f} {lin_az:+6.3f} m/s² | "
              f"Mag: {mag_x:+6.3f} {mag_y:+6.3f} {mag_z:+6.3f} μT | "
              f"Position: X={position[0]:+6.3f} Y={position[1]:+6.3f} Z={position[2]:+6.3f} m")

        # CSV
        if LOG_TO_CSV:
            with open(CSV_FILE, "a", newline='') as f:
                writer = csv.writer(f)
                row = [ts, lin_ax, lin_ay, lin_az, mag_x, mag_y, mag_z, position[0], position[1], position[2]]
                writer.writerow(row)

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

# =============================
# LIVE 3D PLOT SETUP
# =============================
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(-50, 50)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title("Live 3D IMU Position Map (Kalman Filtered)")

ax.scatter([0], [0], [0], color='red', s=150, marker='s', label='Origin')
imu_point = ax.scatter([], [], [], color='blue', s=120, marker='o', label='IMU Position')
trail_line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7)

ax.legend(loc='upper right')
ax.grid(False)
ax.set_box_aspect([1,1,1])

def update(frame):
    fetch_phyphox_data()

    imu_point._offsets3d = ([position[0]], [position[1]], [position[2]])

    if len(position_queue) > 1:
        trail_x = [p[0] for p in position_queue]
        trail_y = [p[1] for p in position_queue]
        trail_z = [p[2] for p in position_queue]
        trail_line.set_data_3d(trail_x, trail_y, trail_z)

    fig.suptitle(f"Live 3D IMU Position (Kalman) | Current: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m")

    return imu_point, trail_line

ani = FuncAnimation(fig, update, interval=POLL_INTERVAL*1000, blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show()