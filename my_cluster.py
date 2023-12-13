from PCAfold import VQPCA
from PCAfold import get_centroids

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.linalg import norm
from scipy.spatial.distance import cdist

class myCluster:
    def __init__(self, data, n_cluster, n_components):
        self.data = data
        self.n_cluster = n_cluster
        self.n_components = n_components
        # Creates self.vqpca attributes wherer the rec. error in min after 10 turns.
        self._initialize_vqpca()
        self.eigenvectors = np.stack(self.vqpca.A)
        self.centroids = get_centroids(data, self.vqpca.idx)
        # Create self.model which is AgglomerativeClustering object.
        self._init_agglo_clustering()
        # Create new eigenvectors based on hierarchy of clusters.
        self._create_hierarchy()

    def _initialize_vqpca(self):
        # Perform VQPCA and safe eigenvectors + centroids
        # Idea: Perform it 10 times and take the representation with the min rec. error
        self.vqpca = VQPCA(self.data, n_clusters=self.n_cluster, n_components=self.n_components,
                            scaling='none', idx_init='random', max_iter=100, tolerance=1.0e-16, verbose=False) 
        for i in range(20):
            try:
                vqpca = VQPCA(self.data, n_clusters=self.n_cluster, n_components=self.n_components,
                            scaling='none', idx_init='random', max_iter=100, tolerance=1.0e-16, verbose=False)
                if(vqpca.global_mean_squared_reconstruction_error < self.vqpca.global_mean_squared_reconstruction_error):
                    self.vqpca = vqpca
                    print(vqpca.global_mean_squared_reconstruction_error)
            except:
                print(f'In iteration {i+1} an error occured!')

    def _init_agglo_clustering(self):
        # try if it is better then 'ward' (based on variance) & eucledian. Make parameters variable.
        # Create hierarchy of clusters. ONLY based on first eigenvectors
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='average',
                                        metric='precomputed')
        frob_array = self.eigenvectors.reshape(self.n_cluster, -1)
        # Compute pairwise Frobenius distances
        distance_matrix = cdist(frob_array, frob_array, metric='euclidean')
        self.model = model.fit(distance_matrix)

    def _create_hierarchy(self):
        self.hierarchy = []
        for eig in range(self.eigenvectors.shape[0]):
            self.hierarchy.append({'eigenvectors': self.eigenvectors[eig, :, :],
                                    'id': eig, 'parents': 'leaf', 'layer': 0,
                                    'clusterPoints': self.data[self.vqpca.idx == eig]})
        self._agg_eigenvectors()

    def _agg_eigenvectors(self):
        # For each child, create new aggregated vector and centroid
        for i, child in enumerate(self.model.children_):
            agg_row = np.expand_dims(np.mean(self.eigenvectors[[child[0], child[1]], :], axis=0), axis=0)
            self.eigenvectors = np.vstack([self.eigenvectors, agg_row])
            desired_ids = [child[0], child[1]]
            desired_layer = [entry['layer'] for entry in self.hierarchy if entry['id'] in desired_ids]
            desired_cluster_points = [entry['clusterPoints'] for entry in self.hierarchy if entry['id'] in desired_ids] 
            self.hierarchy.append({'eigenvectors': np.squeeze(agg_row, axis=0), 'id': i+self.n_cluster,
                                   'parents': (child[0], child[1]), 'layer': max(desired_layer) + 1,
                                   'clusterPoints': np.concatenate(desired_cluster_points, axis=0)})
            agg_cen = np.mean(self.centroids[[child[0], child[1]], :], axis=0).T
            self.centroids = np.vstack([self.centroids, agg_cen])

    def create_principle_components(self, *args):
        principal_components = []
        for i in args:
            for dic in self.hierarchy:
                if(dic['id'] == i):
                    principal_components.append((dic['clusterPoints'] - self.centroids[i]) @ dic['eigenvectors'])
        return principal_components
            
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

