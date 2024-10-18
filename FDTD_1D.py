import matplotlib.pyplot as plt
import numpy as np

# Define constants
jmax = 500
jsource = 100
nmax = 2000
#1D boundary conditions
# Field arrays
Ex = np.zeros(jmax)
Hz = np.zeros(jmax)
Ex_prev = np.zeros(jmax)
Hz_prev = np.zeros(jmax)

lambda_min = 350e-9  # meters
dx = lambda_min / 20
dt = dx / (1 / np.sqrt(8.8541878128e-12 * 1.256637062e-6))  # Speed of light calculation

# Permittivity array (eps)
eps = np.ones(jmax) * 8.8541878128e-12
eps[250:300] = 10 * 8.8541878128e-12  # Higher permittivity region

# Define source function
def Source_Function(t):
    lambda_0 = 550e-9
    w0 = 2 * np.pi * (1 / np.sqrt(8.8541878128e-12 * 1.256637062e-6)) / lambda_0
    tau = 30
    t0 = tau ** 3
    return np.exp(-(t - t0) ** 2 / tau ** 2) * np.sin(w0 * t * dt)

# Main FDTD loop
for n in range(nmax):
    # Update magnetic field boundaries
    Hz[jmax - 1] = Hz_prev[jmax - 2]
    
    # Update magnetic field
    for j in range(jmax - 1):
        Hz[j] = Hz_prev[j] + (dt / (dx * 1.256637062e-6)) * (Ex[j + 1] - Ex[j])
        Hz_prev[j] = Hz[j]
    
    # Magnetic field source
    Hz[jsource - 1] -= Source_Function(n) / np.sqrt(1.256637062e-6 / 8.8541878128e-12)
    Hz_prev[jsource - 1] = Hz[jsource - 1]

    # Update electric field boundaries
    Ex[0] = Ex_prev[1]
    
    # Update electric field
    for j in range(1, jmax):
        Ex[j] = Ex_prev[j] + (dt / (dx * eps[j])) * (Hz[j] - Hz[j - 1])
        Ex_prev[j] = Ex[j]
    
    # Electric field source
    Ex[jsource] += Source_Function(n + 1)
    Ex_prev[jsource] = Ex[jsource]

    # Plot every 10 steps
    if n % 10 == 0:
        plt.plot(Ex)
        plt.plot(eps > 8.8541878128e-12)  # Visualize the material profile
        plt.ylim([-1, 1])
        plt.xlim([0, jmax])
        plt.show()
        plt.close()

