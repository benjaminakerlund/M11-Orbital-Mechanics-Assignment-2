import math


def mean_to_true_anomaly(M_deg, e):
    # Convert mean anomaly to radians
    M = math.radians(M_deg)

    # Initial guess for E (eccentric anomaly)
    E = M if e < 0.8 else math.pi

    # Solve Kepler's equation using Newton-Raphson iteration
    tol = 1e-6
    for _ in range(100):
        f = E - e * math.sin(E) - M
        f_prime = 1 - e * math.cos(E)
        E_next = E - f / f_prime
        if abs(E_next - E) < tol:
            E = E_next
            break
        E = E_next

    # Convert eccentric anomaly to true anomaly
    cos_nu = (math.cos(E) - e) / (1 - e * math.cos(E))
    sin_nu = (math.sqrt(1 - e ** 2) * math.sin(E)) / (1 - e * math.cos(E))

    # Compute the true anomaly in radians and then convert to degrees
    nu = math.atan2(sin_nu, cos_nu)
    nu_deg = math.degrees(nu)

    return nu_deg


# Example usage
mean_anomaly_deg = 68.1381
eccentricity = 0.0006188
true_anomaly_deg = mean_to_true_anomaly(mean_anomaly_deg, eccentricity)
print(f"True Anomaly: {true_anomaly_deg:.4f} degrees")
