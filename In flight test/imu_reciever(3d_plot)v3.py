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
import sys
import signal

# =============================
# CONFIGURATION
# =============================
PHYPH0X_IP = "192.168.100.229"      # Your Phyphox IP
PHYPH0X_PORT = 8080
BASE_URL = f"http://{PHYPH0X_IP}:{PHYPH0X_PORT}/get?"

# Only Acceleration without g
SENSORS = ["accX", "accY", "accZ"]

POLL_INTERVAL = 0.1               # seconds
MAX_TRAIL = 50                    # trail length in live plot
CUBE_SIZE = 50.0                  # ±50 m = 100 × 100 × 100 m total space
LOG_TO_CSV = True
CSV_FILE = "phyphox_acc_log.csv"
HISTORY_FILE = "position_history.csv"  # for static graph

# =============================
# POSITION & HISTORY
# =============================
position = np.array([0.0, 0.0, 0.0])   # current displacement (m)
velocity = np.array([0.0, 0.0, 0.0])   # current velocity (m/s)
dt = POLL_INTERVAL

# History for static graph on exit (time, posX, posY, posZ)
position_history = []  # list of [ts, x, y, z]

still_counter = 0
STILL_THRESHOLD = 0.25            # m/s² threshold for "still"
STILL_FRAMES = 30                 # consecutive frames to confirm still

position_queue = deque(maxlen=MAX_TRAIL)

# Gravity (Z negative when phone flat)
GRAVITY = np.array([0.0, 0.0, +9.81])

# =============================
# CSV SETUP
# =============================
if LOG_TO_CSV:
    with open(CSV_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "accX", "accY", "accZ"])

# History CSV for static graph
with open(HISTORY_FILE, "a", newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(["timestamp", "posX", "posY", "posZ"])

# =============================
# PHYPH0X POLLING
# =============================
def fetch_phyphox_data():
    global position, velocity, still_counter

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

        ax = buffer.get("accX", {}).get("buffer", [0.0])[-1]
        ay = buffer.get("accY", {}).get("buffer", [0.0])[-1]
        az = buffer.get("accZ", {}).get("buffer", [0.0])[-1]

        raw_acc = np.array([ax, ay, az])
        lin_acc = raw_acc - GRAVITY

        acc_mag = np.linalg.norm(lin_acc)

        if acc_mag < STILL_THRESHOLD:
            still_counter += 1
            if still_counter >= STILL_FRAMES:
                velocity = np.array([0.0, 0.0, 0.0])
                print(f"[{ts}] Still detected — velocity reset")
        else:
            still_counter = 0

        velocity += lin_acc * dt
        position += velocity * dt

        position_queue.append(position.copy())

        # Save history for static graph
        position_history.append([ts, position[0], position[1], position[2]])

        print(f"[{ts}] Acc: {ax:+6.3f} {ay:+6.3f} {az:+6.3f} m/s² | "
              f"Pos: X={position[0]:+6.3f} Y={position[1]:+6.3f} Z={position[2]:+6.3f} m")

        if LOG_TO_CSV:
            with open(CSV_FILE, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ts, ax, ay, az])

        with open(HISTORY_FILE, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ts, position[0], position[1], position[2]])

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

# =============================
# LIVE 3D PLOT
# =============================
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-CUBE_SIZE, CUBE_SIZE)
ax.set_ylim(-CUBE_SIZE, CUBE_SIZE)
ax.set_zlim(-CUBE_SIZE, CUBE_SIZE)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title("Live 3D IMU Displacement Map (100 m cube)")

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

    fig.suptitle(f"Live 3D Displacement | Current: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m")

    return imu_point, trail_line

ani = FuncAnimation(fig, update, interval=POLL_INTERVAL*1000, blit=False, cache_frame_data=False)

# =============================
# STATIC GRAPH ON EXIT
# =============================
def show_static_graph():
    if not position_history:
        print("No position history collected.")
        return

    times = [datetime.strptime(row[0], "%H:%M:%S.%f") for row in position_history]
    pos_x = [row[1] for row in position_history]
    pos_y = [row[2] for row in position_history]
    pos_z = [row[3] for row in position_history]

    start_time = times[0]
    seconds = [(t - start_time).total_seconds() for t in times]

    plt.figure(figsize=(12, 6))
    plt.plot(seconds, pos_x, label='X Displacement', color='r')
    plt.plot(seconds, pos_y, label='Y Displacement', color='g')
    plt.plot(seconds, pos_z, label='Z Displacement', color='b')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Displacement (meters)')
    plt.title('Displacement History After Live Session')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("displacement_history_graph.png")
    plt.show()
    print("Static graph saved as 'displacement_history_graph.png'")

def on_exit(sig=None, frame=None):
    print("\nExiting live plot... Generating static displacement graph.")
    show_static_graph()
    sys.exit(0)

signal.signal(signal.SIGINT, on_exit)

plt.tight_layout()
plt.show()

# If window closed normally, still show graph
on_exit()