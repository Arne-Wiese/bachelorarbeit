from PCAfold import VQPCA
import matplotlib.pyplot as plt
import numpy as np

n_points = 100
save_filename = None
global_color = '#454545'
k1_color = '#0e7da7'
k2_color = '#ceca70'

# Parameters for cluster 1
mean_local_1 = [5, 1]
covariance_local_1 = np.array([[0.3, 0.3], [0.3, 0.3]])  # Swap off-diagonal elements
covariance_local_1 = np.dot(covariance_local_1, covariance_local_1.T)

# Parameters for cluster 2
mean_local_2 = [13, 3]
covariance_local_2 = np.array([[0.3, -2], [0, 0.5]])  # Swap off-diagonal elements
covariance_local_2 = np.dot(covariance_local_2, covariance_local_2.T)

# Parameters for cluster 3
mean_local_3 = [8, 8]
covariance_local_3 = np.array([[0.5, 0.4], [0.4, 0.3]])  # Set off-diagonal element to 0
covariance_local_3 = np.dot(covariance_local_3, covariance_local_3.T)

# Generate samples for each cluster
x_noise_1, y_noise_1 = np.random.multivariate_normal(mean_local_1, covariance_local_1, n_points).T
x_noise_2, y_noise_2 = np.random.multivariate_normal(mean_local_2, covariance_local_2, n_points).T
x_noise_3, y_noise_3 = np.random.multivariate_normal(mean_local_3, covariance_local_3, n_points).T

# Concatenate the data points
x_local = np.concatenate([x_noise_1, x_noise_2, x_noise_3])
y_local = np.concatenate([y_noise_1, y_noise_2, y_noise_3])

Dataset_local = np.hstack((x_local[:, np.newaxis], y_local[:, np.newaxis]))

# Instantiate VQPCA class object:
vqpca = VQPCA(
    Dataset_local,
    n_clusters=3,
    n_components=1,
    scaling='std',
    idx_init='random',
    max_iter=100,
    tolerance=1.0e-25,
    verbose=True
)

# Access the VQPCA clustering solution:
idx = vqpca.idx
pcs = vqpca.A
print(pcs)

# Separate data points based on cluster index
cluster_0 = Dataset_local[idx == 0]
cluster_1 = Dataset_local[idx == 1]
cluster_2 = Dataset_local[idx == 2]

# Plot the clusters
plt.scatter(cluster_0[:, 0], cluster_0[:, 1], label='Cluster 0', color='blue')
plt.scatter(cluster_1[:, 0], cluster_1[:, 1], label='Cluster 1', color='orange')
plt.scatter(cluster_2[:, 0], cluster_2[:, 1], label='Cluster 2', color='green')

# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clusters based on Index Vector')

# Show legend
plt.legend()

# Show the plot
plt.show()
