import numpy as np
import matplotlib.pyplot as plt

# Define the softmax function with temperature
def softmax(logits, temperature=1.0):
    exp_values = np.exp(logits / temperature)
    return exp_values / np.sum(exp_values)

# Generate random logits (un-normalized scores) for 10 classes
np.random.seed(42)  # Set seed for reproducibility
logits = 30 + np.random.rand(10) * 10  # Random numbers scaled up to 10

# Define two temperature values
temperature_1 = 1.0  # Normal temperature (standard softmax)
temperature_2 = 0.3  # Lower temperature (sharpens the distribution)

# Calculate softmax distributions
softmax_temp_1 = softmax(logits, temperature_1)
softmax_temp_2 = softmax(logits, temperature_2)

# Plot the distributions using bar plots
plt.figure(figsize=(10, 6))
x = np.arange(len(logits))

# Plot bars for temperature 1
plt.bar(x - 0.15, softmax_temp_1, width=0.3, label=f'Temperature = {temperature_1}', color='blue')

# Plot bars for temperature 2
plt.bar(x + 0.15, softmax_temp_2, width=0.3, label=f'Temperature = {temperature_2}', color='red')

# Add labels and title
#plt.title('Effect of Temperature on Softmax Distribution (Random Logits for 10 Classes)')
plt.xlabel('Token')
plt.ylabel('Probability')
#plt.xticks(x, [f'Class {i+1}' for i in x])
plt.xticks(x, [f'{i+1}' for i in x])
plt.legend()

# Show the plot
plt.grid(True, axis='y')
plt.show()
