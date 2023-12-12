from PCAfold import VQPCA
from PCAfold import get_centroids

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

#TODO: less variables
class myCluster:
    def __init__(self, data, n_cluster, n_components):
        self.data = data
        self.n_cluster = n_cluster
        self.n_components = n_components
        self.vqpca = VQPCA(data, n_clusters=n_cluster, n_components=n_components,
                       scaling='none', idx_init='random', max_iter=100, tolerance=1.0e-08, verbose=False)
        self.eigenvectors = np.stack(self.vqpca.A)
        self.centroids = get_centroids(data, self.vqpca.idx)
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        self.model = model.fit((np.array(self.eigenvectors)[:, :, 0]))
        self.hierarchy = []
        for eig in range(self.eigenvectors.shape[0]):
            self.hierarchy.append({'eigenvectors': self.eigenvectors[eig, :, :],
                                    'id': eig, 'parents': 'leaf', 'layer': 0})

    def plot_dendrogram(self, **kwargs):
        # Create linkage matrix and then plot the dendrogram
        # create the counts of samples under each node
        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)
        for i, merge in enumerate(self.model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        linkage_matrix = np.column_stack(
            [self.model.children_, self.model.distances_, counts]
        ).astype(float)
        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)


    def agg_eigenvectors(self):
        # For each child, create new aggregated vector and centroid
        for i, child in enumerate(self.model.children_):
            agg_row = np.expand_dims(np.mean(self.eigenvectors[[child[0], child[1]], :], axis=0), axis=0)
            self.eigenvectors = np.vstack([self.eigenvectors, agg_row])
            desired_ids = [child[0], child[1]]
            desired_layer = [entry['layer'] for entry in self.hierarchy if entry['id'] in desired_ids]
            self.hierarchy.append({'eigenvectors': np.squeeze(agg_row, axis=0), 'id': i+self.n_cluster,
                                   'parents': (child[0], child[1]), 'layer': max(desired_layer) + 1})
            agg_cen = np.mean(self.centroids[[child[0], child[1]], :], axis=0).T
            self.centroids = np.vstack([self.centroids, agg_cen])

    def plot_2d_scatter(self, dim1, dim2):
        # Number of categories (clusters, groups, etc.)
        n_categories = self.eigenvectors.shape[0]
        # Get the Viridis colormap
        viridis_cmap = plt.cm.get_cmap('inferno', n_categories)
        # Generate a list of colors
        colors = [viridis_cmap(i) for i in range(n_categories)]
        # Access the VQPCA clustering solution:
        for i in range(self.n_cluster):
            plt.scatter(self.data[self.vqpca.idx == i][:, dim1], self.data[self.vqpca.idx == i][:, dim2],
                        color=colors[i], label=f'Cluster {i} ({len(self.data[self.vqpca.idx == i])})')
        
        plt.quiver(self.centroids[:, dim1], self.centroids[:, dim2],
                    self.eigenvectors[:, dim1, 0], self.eigenvectors[:, dim2, 0],
                     color=colors, scale=5)
        
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        # Show the 
        # Show legend
        plt.legend()
        plt.show()

