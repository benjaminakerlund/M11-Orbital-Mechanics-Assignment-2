import matplotlib
matplotlib.use('QtAgg')
import numpy as np
import matplotlib.pyplot as plt


def compute_LOF(omega, theta, i, OMEGA):
    # Defining transformation matrixes
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

    # Compute the overall transformation matrix
    transformation_matrix = R1 @ R2 @ R3

    # Example vector in the Local Orbital Frame (LOF)
    vector_LOF = np.array([1, 1, 1])

    # Transform the vector to the inertial frame
    vector_inertial = transformation_matrix @ vector_LOF

    print("Transformation Matrix:")
    print(transformation_matrix)
    print("\nVector in Inertial Frame:")
    print(vector_inertial)


def computeandplot_LOF(omega, theta, i, OMEGA):
    # Re-Define the transformation matrices
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
    plt.plot(time_steps, lof_vectors[:, 0], label='R Component')
    plt.plot(time_steps, lof_vectors[:, 2], label='W Component')
    plt.plot(time_steps, lof_vectors[:, 1], label='S Component')
    plt.xlabel('Time (radians)')
    plt.ylabel('LOF Vector Components')
    plt.title('LOF Vector Directions Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('../doc/Graphics/LOF_vector_directions_plot')
    plt.show()


# Initial parameters
omega = np.radians(40.8630)
theta = np.radians(68.2039)
i = np.radians(51.6)
OMEGA = np.radians(40.3677)

compute_LOF(omega, theta, i, OMEGA)
computeandplot_LOF(omega, theta, i, OMEGA)


