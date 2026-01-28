import requests
import time

# Replace with your phone's Phyphox IP & port
PHY_PHOX_URL = "http://192.168.101.89/data.json"

def fetch_imu():
    try:
        response = requests.get(PHY_PHOX_URL, timeout=0.5)
        data = response.json()

        # Accelerometer
        ax = data['acceleration']['x']
        ay = data['acceleration']['y']
        az = data['acceleration']['z']

        # Gyroscope
        gx = data['gyroscope']['x']
        gy = data['gyroscope']['y']
        gz = data['gyroscope']['z']

        # Attitude (Euler angles)
        roll = data['attitude']['roll']
        pitch = data['attitude']['pitch']
        yaw = data['attitude']['yaw']

        # Quaternion
        qw = data['quaternion']['w']
        qx = data['quaternion']['x']
        qy = data['quaternion']['y']
        qz = data['quaternion']['z']

        return {
            "acc": (ax, ay, az),
            "gyro": (gx, gy, gz),
            "euler": (roll, pitch, yaw),
            "quat": (qw, qx, qy, qz)
        }

    except Exception as e:
        print("Error fetching IMU:", e)
        return None

# -------------------------
# MAIN LOOP
# -------------------------
while True:
    imu = fetch_imu()
    if imu:
        print(f"Accel: {imu['acc']}")
        print(f"Gyro: {imu['gyro']}")
        print(f"Euler: {imu['euler']}")
        print(f"Quat: {imu['quat']}")
        print("-"*40)
    
    time.sleep(0.05)  # 20 Hz
