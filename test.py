import numpy as np
import matplotlib.pyplot as plt


def displacement_correlation(dt, tau_L):
    dt = np.asarray(dt, dtype=float)
    num = tau_L * (1.0 - np.exp(-dt / tau_L))**2
    den = 2.0 * (dt - tau_L * (1.0 - np.exp(-dt / tau_L)))
    return num / den


def mutual_information_displacements(dt, tau_L):
    rho = displacement_correlation(dt, tau_L)
    rho = np.clip(rho, -1.0 + 1e-15, 1.0 - 1e-15)
    return -0.5 * np.log(1.0 - rho**2)


# parameters
tau_L = 2.0
dt_vals = np.linspace(1e-3, 3.0 * tau_L, 500)

# compute MI
mi_vals = mutual_information_displacements(dt_vals, tau_L)

mi_vel_vals = -0.5 * np.log(1.0 - np.exp(-2.0 * dt_vals / tau_L))

# plot
plt.figure(figsize=(6, 4))
plt.plot(dt_vals, mi_vals, label=r'Displacement MI')
plt.plot(dt_vals, mi_vel_vals, '--', label=r'Velocity MI')
plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$I$')
plt.title('Mutual Information vs Time Lag')
plt.legend()
plt.tight_layout()
plt.show()