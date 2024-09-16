# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Create 2D random data from two mixtures of Gaussians
n_samples = 300
true_centers = [(-2, -2), (2, 2)]  # True centers of the Gaussians
cluster_std = [1.0, 1.5]           # True standard deviations
X, y = make_blobs(n_samples=n_samples, centers=true_centers, cluster_std=cluster_std, random_state=42)

# Plotting the data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.title('Random 2D data from two distinct classes')
plt.show()

# Part 1: Discriminative Learning - SVM Classifier
# ----------------------
# Train an SVM classifier
svm = SVC(kernel='linear', probability=True)
svm.fit(X, y)

# Identify the support vectors
support_vectors = svm.support_vectors_

# Create a meshgrid for plotting decision boundaries
h = .02  # Step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on meshgrid
Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)

# Plot SVM decision boundary
plt.contourf(xx, yy, Z_svm, alpha=0.8, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')

# Highlight the support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
            facecolors='none', edgecolors='yellow', s=100, label='Support Vectors')

plt.title('SVM Decision Boundaries with Support Vectors')
plt.legend()
plt.show()

# Part 2:  Generative Learning - GMM Classifier
# ----------------------
# Train a GMM classifier
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict on meshgrid for GMM
probs = gmm.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z_gmm = probs[:, 1] > 0.5  # Using 0.5 as a threshold
Z_gmm = Z_gmm.reshape(xx.shape)

# Plot GMM decision boundary
plt.contourf(xx, yy, Z_gmm, alpha=0.8, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
plt.title('GMM Decision Boundaries')
plt.show()

# Compare the GMM Gaussians with the true Gaussians
print("\nTrue Gaussian Parameters:")
print("Centers:", true_centers)
print("Standard Deviations:", cluster_std)

print("\nGMM Estimated Parameters:")
for i in range(gmm.n_components):
    print(f"Gaussian {i+1}:")
    print("Mean:", gmm.means_[i])
    print("Covariance Matrix:\n", gmm.covariances_[i])
    print()

###################

# Separate the points by class
X_class0 = X[y == 0]
X_class1 = X[y == 1]

# Create 2D normalized histograms for both classes
x_edges = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 50)
y_edges = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 50)
hist_class0, x_edges, y_edges = np.histogram2d(X_class0[:, 0], X_class0[:, 1], bins=(x_edges, y_edges), density=True)
hist_class1, x_edges, y_edges = np.histogram2d(X_class1[:, 0], X_class1[:, 1], bins=(x_edges, y_edges), density=True)

# Create the X and Y meshgrid for plotting
x_mid = (x_edges[:-1] + x_edges[1:]) / 2
y_mid = (y_edges[:-1] + y_edges[1:]) / 2
X_grid, Y_grid = np.meshgrid(x_mid, y_mid)

# Create 3D plot of the 2D histograms for both classes
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot histogram for class 0 as a 3D surface
ax.plot_surface(X_grid, Y_grid, hist_class0.T, cmap='Blues', edgecolor='none', alpha=0.6, label='Class 0 Histogram')

# Plot histogram for class 1 as a 3D surface
ax.plot_surface(X_grid, Y_grid, hist_class1.T, cmap='Reds', edgecolor='none', alpha=0.6, label='Class 1 Histogram')

# Now let's calculate the PDF for each GMM component

# Flatten the grid to input into GMM's score_samples method
x_flat = np.c_[X_grid.ravel(), Y_grid.ravel()]

# For GMM, get the parameters for each component (mean and covariance)
gmm_means = gmm.means_
gmm_covariances = gmm.covariances_

# Evaluate PDFs for each Gaussian component
gmm_pdf_class0 = multivariate_normal(mean=gmm_means[0], cov=gmm_covariances[0]).pdf(x_flat).reshape(X_grid.shape)
gmm_pdf_class1 = multivariate_normal(mean=gmm_means[1], cov=gmm_covariances[1]).pdf(x_flat).reshape(X_grid.shape)

# Plot the GMM PDF for class 0
ax.plot_surface(X_grid, Y_grid, gmm_pdf_class0, cmap='cool', edgecolor='none', alpha=0.4, label='Class 0 GMM PDF')

# Plot the GMM PDF for class 1
ax.plot_surface(X_grid, Y_grid, gmm_pdf_class1, cmap='autumn', edgecolor='none', alpha=0.4, label='Class 1 GMM PDF')

# Set labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Density')
ax.set_title('3D Plot of 2D Histogram with GMM PDFs for Each Class')

plt.show()
