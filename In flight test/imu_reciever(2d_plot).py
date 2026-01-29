import requests
import json
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

# =============================
# CONFIGURATION
# =============================
PHYPH0X_IP = "192.168.100.229"      # Your current Phyphox IP
PHYPH0X_PORT = 8080
BASE_URL = f"http://{PHYPH0X_IP}:{PHYPH0X_PORT}/get?"

SENSORS = [
    "accX", "accY", "accZ",         # Accelerometer
    "gyrX", "gyrY", "gyrZ",         # Gyroscope
    "linAccX", "linAccY", "linAccZ" # Linear Acceleration
]

POLL_INTERVAL = 0.1               # seconds
MAX_POINTS = 100                  # How many points to show in plot
LOG_TO_CSV = True
CSV_FILE = "phyphox_imu_log.csv"

# =============================
# DATA STORAGE
# =============================
data_queues = {s: deque(maxlen=MAX_POINTS) for s in SENSORS}
timestamps = deque(maxlen=MAX_POINTS)

# =============================
# CSV SETUP
# =============================
if LOG_TO_CSV:
    with open(CSV_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp"] + SENSORS)

# =============================
# PHYPH0X POLLING FUNCTION
# =============================
def fetch_phyphox_data():
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
        new_values = {}
        for sensor in SENSORS:
            if sensor in buffer and buffer[sensor]["buffer"]:
                val = buffer[sensor]["buffer"][-1]
                new_values[sensor] = val
                data_queues[sensor].append(val)
            else:
                new_values[sensor] = 0.0
                data_queues[sensor].append(0.0)

        timestamps.append(ts)

        # Print
        print(f"[{ts}] "
              f"Acc: {new_values.get('accX',0):+6.3f} {new_values.get('accY',0):+6.3f} {new_values.get('accZ',0):+6.3f} | "
              f"Gyro: {new_values.get('gyrX',0):+7.2f} {new_values.get('gyrY',0):+7.2f} {new_values.get('gyrZ',0):+7.2f} | "
              f"LinAcc: {new_values.get('linAccX',0):+6.3f} {new_values.get('linAccY',0):+6.3f} {new_values.get('linAccZ',0):+6.3f}")

        # CSV
        if LOG_TO_CSV:
            with open(CSV_FILE, "a", newline='') as f:
                writer = csv.writer(f)
                row = [ts] + [new_values.get(s, 0.0) for s in SENSORS]
                writer.writerow(row)

        return True

    except Exception as e:
        print(f"Error fetching data: {e}")
        return False

# =============================
# LIVE PLOT SETUP
# =============================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Phyphox Live IMU Data", fontsize=16)

lines = {}
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
for i, sensor in enumerate(SENSORS):
    if "acc" in sensor:
        line, = ax1.plot([], [], color=colors[i % len(colors)], label=sensor)
    else:
        line, = ax2.plot([], [], color=colors[i % len(colors)], label=sensor)
    lines[sensor] = line

ax1.set_title("Accelerometer & Linear Acceleration (m/s²)")
ax1.set_ylabel("Acceleration")
ax1.legend(loc="upper left")
ax1.grid(True)

ax2.set_title("Gyroscope (°/s)")
ax2.set_ylabel("Angular Velocity")
ax2.set_xlabel("Time (samples)")
ax2.legend(loc="upper left")
ax2.grid(True)

def init():
    for line in lines.values():
        line.set_data([], [])
    return list(lines.values())

def animate(i):
    fetch_phyphox_data()  # Get new data

    x = list(range(len(timestamps)))
    for sensor, line in lines.items():
        y = list(data_queues[sensor])
        line.set_data(x, y)

    # Auto-scale y-axis
    for ax in [ax1, ax2]:
        ax.relim()
        ax.autoscale_view()

    return list(lines.values())

ani = animation.FuncAnimation(fig, animate, init_func=init, interval=200, blit=False)

plt.tight_layout()
plt.show()