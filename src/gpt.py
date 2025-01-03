import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')

def compute_orbital_frame(r, v):
    """
    Compute the local orbital frame directions for a CubeSat using RWS frame.
    (And normalise to unit vectors)
    :param r: Position vector in ECI frame (3-element array)
    :param v: Velocity vector in ECI frame (3-element array)
    :return: Unit vectors (R_hat, W_hat, S_hat) in ECI frame
    """
    R_hat = r / np.linalg.norm(r)
    w = np.cross(r, v)              # radial component
    W_hat = w / np.linalg.norm(w)   # orbit normal component
    S_hat = v / np.linalg.norm(v)   # Tangential component
    return R_hat, W_hat, S_hat

# LOF unit vector directions example:
orbital_radius = 6378 + 408
orbital_speed = 7.66
r1 = np.array([orbital_radius, 0, 0])   # Position in km
v1 = np.array([0, orbital_speed, 0])    # Velocity in km/s (ISS)

R_hat, W_hat, S_hat = compute_orbital_frame(r1, v1)
print("Radial Direction (R_hat):", R_hat)
print("Normal Direction (W_hat):", W_hat)
print("Tangential Direction (S_hat):", S_hat)


# Parameters for a circular orbit
# orbital_radius
# orbital_speed
orbital_period = 2 * np.pi * orbital_radius / orbital_speed  # seconds

time_steps = 500
time = np.linspace(0, orbital_period, time_steps)

R_vectors, W_vectors, S_vectors = [], [], []

# Simulate the orbit
for t in time:
    angle = 2 * np.pi * t / orbital_period
    r = np.array([orbital_radius * np.cos(angle), orbital_radius * np.sin(angle), 0])
    v = np.array([-orbital_speed * np.sin(angle), orbital_speed * np.cos(angle), 0])
    R_hat, W_hat, S_hat = compute_orbital_frame(r, v)
    R_vectors.append(R_hat)
    W_vectors.append(W_hat)
    S_vectors.append(S_hat)

R_vectors = np.array(R_vectors)
W_vectors = np.array(W_vectors)
S_vectors = np.array(S_vectors)

# Plot the components over time
plt.figure(figsize=(12, 8))

# Radial direction components
plt.subplot(3, 1, 1)
plt.plot(time / 3600, R_vectors[:, 0], label="R_x", color="r")
plt.plot(time / 3600, R_vectors[:, 1], label="R_y", color="g")
plt.plot(time / 3600, R_vectors[:, 2], label="R_z", color="b")
plt.title("Radial Direction Components Over Time")
plt.xlabel("Time (hours)")
plt.ylabel("Unit Vector Components")
plt.legend()

# Normal direction components
plt.subplot(3, 1, 2)
plt.plot(time / 3600, W_vectors[:, 0], label="W_x", color="r")
plt.plot(time / 3600, W_vectors[:, 1], label="W_y", color="g")
plt.plot(time / 3600, W_vectors[:, 2], label="W_z", color="b")
plt.title("Normal Direction Components Over Time")
plt.xlabel("Time (hours)")
plt.ylabel("Unit Vector Components")
plt.legend()

# Tangential direction components
plt.subplot(3, 1, 3)
plt.plot(time / 3600, S_vectors[:, 0], label="S_x", color="r")
plt.plot(time / 3600, S_vectors[:, 1], label="S_y", color="g")
plt.plot(time / 3600, S_vectors[:, 2], label="S_z", color="b")
plt.title("Tangential Direction Components Over Time")
plt.xlabel("Time (hours)")
plt.ylabel("Unit Vector Components")
plt.legend()

plt.tight_layout()
plt.show()