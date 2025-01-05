import matplotlib
matplotlib.use('QtAgg')
import numpy as np
import matplotlib.pyplot as plt

# Convert degrees to radians for calculations
omega = np.radians(40.8630)
theta = np.radians(68.2039)
i = np.radians(51.6)
OMEGA = np.radians(40.3677)

# Define the transformation matrices
R3_omega_theta = lambda t: np.array([
    [np.cos(omega + t), np.sin(omega + t), 0],
    [-np.sin(omega + t), np.cos(omega + t), 0],
    [0, 0, 1]
])

R1_i = np.array([
    [1, 0, 0],
    [0, np.cos(i), np.sin(i)],
    [0, -np.sin(i), np.cos(i)]
])

R3_OMEGA = np.array([
    [np.cos(OMEGA), np.sin(OMEGA), 0],
    [-np.sin(OMEGA), np.cos(OMEGA), 0],
    [0, 0, 1]
])

# Generate time values and calculate LOF vectors over time
time_steps = np.linspace(0, 2 * np.pi, 100)
lof_vectors = []

for t in time_steps:
    transformation_matrix = R3_omega_theta(t) @ R1_i @ R3_OMEGA
    vector_LOF = np.array([1, 1, 1])
    vector_inertial = transformation_matrix @ vector_LOF
    lof_vectors.append(vector_inertial)

lof_vectors = np.array(lof_vectors)

# Plotting the LOF vector components over time
plt.figure(figsize=(10, 6))
plt.plot(time_steps, lof_vectors[:, 0], label='X Component')
plt.plot(time_steps, lof_vectors[:, 1], label='Y Component')
plt.plot(time_steps, lof_vectors[:, 2], label='Z Component')
plt.xlabel('Time (radians)')
plt.ylabel('LOF Vector Components')
plt.title('LOF Vector Directions Over Time')
plt.legend()
plt.grid(True)
plt.show()

