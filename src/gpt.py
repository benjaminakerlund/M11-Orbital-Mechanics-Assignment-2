import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')


def normalize(vector):
    """Return a normalized vector."""
    return vector / np.linalg.norm(vector)

def compute_orbital_frame(r, v):
    """
    Compute the local orbital frame directions for a CubeSat using RWS frame.
    :param r: Position vector in ECI frame (3-element array)
    :param v: Velocity vector in ECI frame (3-element array)
    :return: Unit vectors (R_hat, W_hat, S_hat) in ECI frame
    """
    R_hat = normalize(r)
    h = np.cross(r, v)
    W_hat = normalize(h)
    S_hat = normalize(np.cross(W_hat, R_hat))
    return R_hat, W_hat, S_hat

def plot_ground_track(inclination_deg, altitude_km, time_steps=500):
    """Plot the ground track of a satellite for a given inclination and altitude."""
    Earth_radius = 6371  # km
    orbital_radius = Earth_radius + altitude_km
    mu = 398600.4418  # Earth’s gravitational parameter (km^3/s^2)
    orbital_speed = np.sqrt(mu / orbital_radius)
    orbital_period = 2 * np.pi * np.sqrt(orbital_radius**3 / mu)
    time = np.linspace(0, orbital_period, time_steps)
    incl_rad = np.radians(inclination_deg)

    latitudes = []
    longitudes = []

    for t in time:
        angle = 2 * np.pi * t / orbital_period
        r = np.array([orbital_radius * np.cos(angle), orbital_radius * np.sin(angle), 0])
        v = np.array([-orbital_speed * np.sin(angle), orbital_speed * np.cos(angle), 0])

        R_hat, W_hat, S_hat = compute_orbital_frame(r, v)

        lat = np.degrees(np.arcsin(np.sin(incl_rad) * np.sin(angle)))
        lon = np.degrees(angle % (2 * np.pi)) - 180

        latitudes.append(lat)
        longitudes.append(lon)

    plt.plot(longitudes, latitudes, label=f"Inclination {inclination_deg}°")

# Plotting ground tracks for different orbits
plt.figure(figsize=(12, 8))
plot_ground_track(0, 408)
plot_ground_track(51.6, 408)
plot_ground_track(80, 408)
plot_ground_track(5, 408)
plot_ground_track(90, 408)

plt.scatter(0, 0, color='red', label='Point P (0°, 0°)', s=100)
plt.title("Ground Tracks for Various Orbits")
plt.xlabel("Longitude (degrees)")
plt.ylabel("Latitude (degrees)")
plt.legend()
plt.grid(True)
plt.show()
