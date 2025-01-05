import matplotlib
matplotlib.use('QtAgg')
import numpy as np
import matplotlib.pyplot as plt
import math
import pytz
from dateutil.relativedelta import relativedelta
from skyfield.api import load, wgs84, EarthSatellite
from datetime import timedelta

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

def calculate_max_view_angle(h):
    # Earth parameters
    R_E = 6378  # Radius of Earth in km

    # Calculate alpha using the formula
    D_H = math.sqrt(2 * R_E * h)
    alpha = math.asin(D_H / (R_E + h))

    # Convert alpha from radians to degrees
    alpha_deg = math.degrees(alpha)

    # Display the result
    print(f"\nAlpha (in radians): {alpha:.6f}")
    print(f"Alpha (in degrees): {alpha_deg:.6f}")

def plot_groundtrack():
    '''
    Parts of this software was reused from homework1
    Ground track computation
    * get starting time from epoch
    * set timescale and calculate subpoints (latitude and longitude)
    * Plot the groundtrack onto an existing map
        * image credits: https://upload.wikimedia.org/wikipedia/commons/2/23/Blue_Marble_2002.png
    '''
    timescale = load.timescale()

    # two-line elements taken from https://celestrak.org/Norad/elements/table.php?GROUP=stations&FORMAT=tle
    # taken on jan 5th
    line1 = '1 25544U 98067A   25005.21659662  .00022613  00000+0  39198-3 0  9995'
    line2 = '2 25544  51.6390  34.3594 0006245  46.9928  71.6484 15.50776974489887'
    satellite = EarthSatellite(line1, line2, 'ISS (ZARYA)', timescale)

    # Orbital elements
    eccentricity = line2[26:33]
    inclination = line2[8:16]
    right_ascension = line2[17:25]
    argument = line2[34:42]
    mean_anomaly = line2[43:51]
    mean_motion = line2[52:63]

    # Calculate semi-major axis
    T = 24 * 60 * 60 / float(mean_motion)
    mu = 3.986 * 10 ** 14
    a = (((T ** 2) * mu) / (4 * (math.pi ** 2))) ** (1 / 3)

    # Get current time in a timezone-aware fashion from the epoch
    tz = pytz.timezone('UTC')
    dt = satellite.epoch.astimezone(tz)
    print()
    print(satellite)
    print(f"Exectution time: {dt:%Y-%m-%d %H:%M:%S %Z}\n")

    # Split 3 orbits (3*92.94061233 minutes) into 400 evenly spaced Timescales as indicated by points
    # 400 chosen for map visibility reasons, could be 101 ==> one plot every 2 min + endpoints
    orbits_min = 100 * T / 60
    t0 = timescale.utc(dt)
    t1 = timescale.utc(dt + relativedelta(minutes=orbits_min))
    timescales = timescale.linspace(t0, t1, 10000)

    # calculate the latitude and longitude subpoints.
    geocentrics = satellite.at(timescales)
    subpoints = wgs84.subpoint_of(geocentrics)
    latitude = subpoints.latitude.degrees
    longitude = subpoints.longitude.degrees

    latitude_filtered = []
    longitude_filtered = []

    # Filter through groundtrack points
    alpha = 19.644385
    for p in range(len(latitude)):
        '''if latitude[p] > -alpha and latitude[p] < alpha:
            if longitude[p] > -alpha and longitude[p] < alpha:
                latitude_filtered.append(latitude[p])
                longitude_filtered.append(longitude[p])
        '''
        if math.sqrt(
            (latitude[p])**2 + (longitude[p])**2
        ) <= math.sqrt(
            (alpha)**2 + (alpha)**2
        ):
            latitude_filtered.append(latitude[p])
            longitude_filtered.append(longitude[p])




    # Load background image
    background_image_path = r'C:\Users\benja\PycharmProjects\groundtrack\earth.jpg'
    background_img = plt.imread(background_image_path)

    # Create the plot
    plt.figure(figsize=(15.2, 8.2))
    plt.imshow(background_img, extent=[-180, 180, -90, 90])

    # Plot the ground track
    title = f"Points of orbit ground track where point P is visible from CubeSat\n" \
            f" Data from: {dt:%Y-%m-%d %H:%M:%S %Z}" #TODO change this
    #plt.scatter(longitude, latitude, label="ISS ground-track", color='red', marker='o', s=1)
    plt.scatter(longitude_filtered, latitude_filtered, label="ISS ground-track", color='blue', marker='o', s=1)
    plt.xlabel("Longitude (degrees, \N{DEGREE SIGN})")
    plt.ylabel("Latitude (degrees, \N{DEGREE SIGN})")
    plt.title(title)

    # Show the plot
    plt.legend()
    plt.grid(True, color='w', linestyle=":", alpha=0.4)
    plt.savefig("../doc/Graphics/P_visible_groundtrack")
    plt.show()

# Initial parameters
omega = np.radians(40.8630)
theta = np.radians(68.2039)
i = np.radians(51.6)
OMEGA = np.radians(40.3677)
h = 408

compute_LOF(omega, theta, i, OMEGA)
if input("Do you want to plot LOF directions [y/n]?") == "y":
    computeandplot_LOF(omega, theta, i, OMEGA)
calculate_max_view_angle(h)
if input("Do you want to plot groundtrack [y/n]?") == "y":
    plot_groundtrack()



