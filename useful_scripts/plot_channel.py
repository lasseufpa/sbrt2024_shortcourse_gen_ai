import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

# Parameters for time and frequency
time = np.linspace(0, 10, 150)  # Time from 0 to 10 seconds (Ns: Symbols)
frequency = np.linspace(0, 300, 150)  # Frequency from 0 to 300 MHz (Nr: Frequency)

# Create a 2D grid of time and frequency
T, F = np.meshgrid(time, frequency)

# Generate a wireless channel response with smooth variations over time and frequency
channel_magnitude = np.abs(np.sin(8 * np.pi * F * 0.05) * np.cos(8 * np.pi * T * 0.05) + 
                           8 * np.random.randn(*T.shape))

# Smooth the surface for a more structured look
smoothed_channel_magnitude = gaussian_filter(channel_magnitude, sigma=5) - 5

print("min=", np.min(smoothed_channel_magnitude))
print("max=", np.max(smoothed_channel_magnitude))


# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with a jet colormap and remove edges for a smoother look
surf = ax.plot_surface(T, F, smoothed_channel_magnitude, cmap='jet', edgecolor='none')

# Adjust the viewpoint to resemble the uploaded image
ax.view_init(elev=30, azim=-120)

# Labels for the axes (using Ns for Symbols and Nr for Frequency)
ax.set_xlabel('Frequency f (MHz)')
ax.set_ylabel('Time t (ms)')
ax.set_zlabel('Magnitude |H(t,f)|')

# Set ticks for a similar appearance
ax.set_xticks(np.linspace(0, 10, 6))  # Time in seconds (Symbols)
ax.set_yticks(np.linspace(0, 300, 6))  # Frequency in MHz
ax.set_zticks(np.linspace(0, 2.5, 6))  # Magnitude

# Add a color bar to represent the magnitude
#fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Channel Magnitude')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


#plt.title('Wireless Channel Magnitude over Frequency and Symbols')
plt.show()
