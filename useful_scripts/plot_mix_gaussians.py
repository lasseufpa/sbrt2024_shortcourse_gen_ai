import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define parameters for the two Gaussians in each mixture
# For the first mixture (PDF 1)
mean1_1, std_dev1_1 = 2, 0.8   # Mean and std_dev for the first Gaussian in the first mixture
mean1_2, std_dev1_2 = 4, 0.5 # Mean and std_dev for the second Gaussian in the first mixture
weight1_1, weight1_2 = 0.6, 0.4  # Weights for the two Gaussians in the first mixture

# For the second mixture (PDF 2)
mean2_1, std_dev2_1 = 7, 0.3   # Mean and std_dev for the first Gaussian in the second mixture
mean2_2, std_dev2_2 = 8, 0.4 # Mean and std_dev for the second Gaussian in the second mixture
weight2_1, weight2_2 = 0.5, 0.5  # Weights for the two Gaussians in the second mixture

# Generate a range of x values
x = np.linspace(0, 10, 500)

# Calculate the PDF for the first mixture of Gaussians
pdf1_1 = norm.pdf(x, mean1_1, std_dev1_1)
pdf1_2 = norm.pdf(x, mean1_2, std_dev1_2)
pdf1 = weight1_1 * pdf1_1 + weight1_2 * pdf1_2

# Calculate the PDF for the second mixture of Gaussians
pdf2_1 = norm.pdf(x, mean2_1, std_dev2_1)
pdf2_2 = norm.pdf(x, mean2_2, std_dev2_2)
pdf2 = weight2_1 * pdf2_1 + weight2_2 * pdf2_2

# Plot the PDFs
plt.plot(x, pdf1, label='Mixture 1', color='red')
plt.plot(x, pdf2, label='Mixture 2', color='blue')

# Add labels and a legend
plt.xlabel('x')
plt.ylabel('Likelihood')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
