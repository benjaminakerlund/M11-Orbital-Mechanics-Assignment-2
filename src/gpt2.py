import matplotlib
matplotlib.use('QtAgg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_LOF_vectors(omega, theta, i, OMEGA):
    R1 = np.array([
        [np.cos(omega + theta), np.sin(omega + theta), 0],
        [-np.sin(omega + theta), np.cos(omega + theta), 0],
        [0, 0, 1]
    ])

    R2 = np.array([
        [1, 0, 0],
        [0, np.cos(i), np.sin(i)],
        [0, -np.sin(i), np.cos(i)]
    ])

    R3 = np.array([
        [np.cos(OMEGA), np.sin(OMEGA), 0],
        [-np.sin(OMEGA), np.cos(OMEGA), 0],
        [0, 0, 1]
    ])

    transformation_matrix = R1 @ R2 @ R3

    # LOF basis vectors
    x_LOF = transformation_matrix @ np.array([0.01, 0, 0])
    y_LOF = transformation_matrix @ np.array([0, 0.01, 0])
    z_LOF = transformation_matrix @ np.array([0, 0, 0.01])

    return x_LOF, y_LOF, z_LOF

# Initialize parameters
omega = np.radians(40.8630)
i = np.radians(51.6)
OMEGA = np.radians(40.3677)

# Time vector over one orbit (simplified to 10 points)
theta_values = np.linspace(0, 2 * np.pi, 10)
x_values, y_values, z_values = [], [], []

# Compute LOF vectors over one orbit
for theta in theta_values:
    x_LOF, y_LOF, z_LOF = compute_LOF_vectors(omega, theta, i, OMEGA)
    x_values.append(x_LOF)
    y_values.append(y_LOF)
    z_values.append(z_LOF)

# Convert to arrays for easier plotting
x_values = np.array(x_values)
y_values = np.array(y_values)
z_values = np.array(z_values)

# Plotting the LOF vectors in 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each LOF basis vector as a line
ax.quiver(0, 0, 0, x_values[:, 0], x_values[:, 1], x_values[:, 2], color='r', label='X_LOF')
ax.quiver(0, 0, 0, y_values[:, 0], y_values[:, 1], y_values[:, 2], color='g', label='Y_LOF')
ax.quiver(0, 0, 0, z_values[:, 0], z_values[:, 1], z_values[:, 2], color='b', label='Z_LOF')


# Set labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Local Orbital Frame (LOF) Vector Directions Over One Orbit')
ax.legend()
plt.show()




