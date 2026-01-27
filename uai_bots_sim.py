import pybullet as p
import pybullet_data
import time
import numpy as np

# ------------------------
# Connect to PyBullet
# ------------------------
physicsClient = p.connect(p.GUI)  # GUI mode
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # for plane.urdf
p.setGravity(0, 0, -9.81)

# ------------------------
# Load environment
# ------------------------
planeId = p.loadURDF("plane.urdf")

# Drone: simple box with 4 motor positions
drone_start_pos = [0, 0, 0.2]
drone_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
droneId = p.loadURDF("r2d2.urdf", drone_start_pos, drone_start_orientation, globalScaling=0.2)

# Motor positions relative to drone CoM (front-left, front-right, back-left, back-right)
motor_offsets = np.array([[0.1, 0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [-0.1, -0.1, 0]])

# ------------------------
# Drone parameters
# ------------------------
mass = 1.0  # kg
k_thrust = 1.0  # simple thrust coefficient
dt = 1/240.0

# PID gains for hover
Kp = np.array([2.0, 2.0, 20.0])  # X, Y, Z
Kd = np.array([0.5, 0.5, 5.0])
Ki = np.array([0.0, 0.0, 0.0])
target_pos = np.array([0, 0, 1.0])
integral_error = np.zeros(3)

# ------------------------
# Simulation loop
# ------------------------
for i in range(10000):
    # Read current state
    pos, orn = p.getBasePositionAndOrientation(droneId)
    lin_vel, ang_vel = p.getBaseVelocity(droneId)
    pos = np.array(pos)
    lin_vel = np.array(lin_vel)

    # PID control for position (hover)
    error = target_pos - pos
    integral_error += error * dt
    derivative = -lin_vel
    force = Kp*error + Kd*derivative + Ki*integral_error
    force[2] += mass * 9.81  # gravity compensation

    # Apply total upward force equally to all 4 motors
    for m_offset in motor_offsets:
        motor_pos_world = pos + m_offset
        p.applyExternalForce(droneId, -1, [0,0,force[2]/4], motor_pos_world, p.WORLD_FRAME)

    p.stepSimulation()
    time.sleep(dt)
