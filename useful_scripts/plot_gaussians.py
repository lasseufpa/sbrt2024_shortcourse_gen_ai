import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define parameters for the two Gaussians
mean1, std_dev1 = 4, 2  # Mean and standard deviation for N(4, 2)
mean2, std_dev2 = 7, 1  # Mean and standard deviation for N(7, 1)

# Generate a range of x values
x = np.linspace(0, 10, 500)

# Calculate the PDF for both Gaussians
pdf1 = norm.pdf(x, mean1, std_dev1)
pdf2 = norm.pdf(x, mean2, std_dev2)

# Plot the PDFs
plt.plot(x, pdf1, label='N(4, 2)', color='red')
plt.plot(x, pdf2, label='N(7, 1)', color='blue')

# Add labels and a legend
# plt.title('PDF of N(4, 2) and N(7, 1)')
plt.xlabel('x')
plt.ylabel('Likelihood')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
