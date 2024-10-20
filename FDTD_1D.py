import matplotlib.pyplot as plt
import numpy as np

# Define constants
jmax = 500           # Number of spatial points
jsource = 100        # Source location
nmax = 3000          # Number of time steps (increase to see the wave propagate)
c0 = 3e8             # Speed of light in vacuum

# Field arrays (Electric and Magnetic fields)
Ex = np.zeros(jmax)       # Electric field
Hz = np.zeros(jmax)       # Magnetic field
Ex_prev = np.zeros(jmax)  # Previous timestep electric field
Hz_prev = np.zeros(jmax)  # Previous timestep magnetic field

# Define spatial and temporal steps
lambda_min = 550e-9      # Wavelength of the sine wave in meters (match to the source frequency)
dx = lambda_min / 20     # Spatial step
dt = dx / c0             # Time step based on the speed of light

# Define permittivity (eps) - higher in a region to represent a dielectric
eps0 = 8.8541878128e-12  # Permittivity of free space
eps = np.ones(jmax) * eps0  # Default permittivity (free space)
eps[250:300] = 10 * eps0    # Higher permittivity region (dielectric)

# Define a sinewave source function with a frequency corresponding to lambda_min
def Source_Function(t):
    freq = c0 / lambda_min  # Frequency based on wavelength
    return np.sin(2 * np.pi * freq * t * dt)  # Sine wave source

# Main FDTD loop
for n in range(nmax):
    # Update magnetic field boundaries (open boundary)
    Hz[jmax - 1] = Hz_prev[jmax - 2]
    
    # Update magnetic field using FDTD equations
    for j in range(jmax - 1):
        Hz[j] = Hz_prev[j] + (dt / (dx * 1.256637062e-6)) * (Ex[j + 1] - Ex[j])
    
    # Magnetic field source (subtract from Hz at the source)
    Hz[jsource - 1] -= Source_Function(n) / np.sqrt(1.256637062e-6 / eps0)
    
    # Update the previous magnetic field
    Hz_prev = Hz.copy()

    # Update electric field boundaries (open boundary)
    Ex[0] = Ex_prev[1]
    
    # Update electric field using FDTD equations
    for j in range(1, jmax):
        Ex[j] = Ex_prev[j] + (dt / (dx * eps[j])) * (Hz[j] - Hz[j - 1])
    
    # Electric field source (add to Ex at the source)
    Ex[jsource] += Source_Function(n + 1)
    
    # Update the previous electric field
    Ex_prev = Ex.copy()

    # Plot the electric field and the dielectric material profile every 50 steps
    if n % 50 == 0:
        plt.plot(Ex, label="Electric Field Ex")
        plt.plot(eps > eps0, color='orange', label="Dielectric Region")  # Visualize the dielectric
        plt.ylim([-1.2, 1.2])
        plt.xlim([0, jmax])
        plt.title(f"Time step: {n}")
        plt.legend()
        plt.pause(0.1)  # Pause to allow the plot to be displayed
        plt.clf()       # Clear the figure for the next plot

plt.show()
