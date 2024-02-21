from PCAfold import VQPCA
from PCAfold import get_centroids

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.linalg import norm
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import sys
import matplotlib.tri as mtri
import itertools
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition


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
        while True:
            try:
                self.vqpca = VQPCA(self.data, n_clusters=self.n_cluster, n_components=self.n_components,
                            scaling='none', idx_init='random', max_iter=100, tolerance=1.0e-16, verbose=False)
                break
            except:
                pass
        for _ in range(15):
            while True:
                try:
                    vqpca = VQPCA(self.data, n_clusters=self.n_cluster, n_components=self.n_components,
                                scaling='none', idx_init='random', max_iter=100, tolerance=1.0e-20, verbose=False)
                    if (vqpca.global_mean_squared_reconstruction_error < self.vqpca.global_mean_squared_reconstruction_error):
                        self.vqpca = vqpca
                    break
                except:
                    pass

    def _init_agglo_clustering(self):
        # try if it is better then 'ward' (based on variance) & eucledian. Make parameters variable.
        # Create hierarchy of clusters. ONLY based on first eigenvectors
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='average',
                                        metric='precomputed')
        # Reshape array from (x, y, n) to (x, y)
        frob_array = self.eigenvectors.reshape(self.n_cluster, -1)
        # Compute pairwise Frobenius distances
        distance_matrix = cdist(frob_array, frob_array, 'euclidean')
        distance_matrix1 = cdist(frob_array, -frob_array, 'euclidean')
        # Get the min dist entry of both matrices
        min_matrix = np.minimum(distance_matrix, distance_matrix1)
        self.model = model.fit(min_matrix)

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
            agg_row = np.mean(self.eigenvectors[[child[0], child[1]], :], axis=0)
            agg_row1 = np.mean(self.eigenvectors[[child[0], -child[1]], :], axis=0)
            # Compute the frobenius distance for both directions
            euclidean_distances = np.linalg.norm(agg_row - self.eigenvectors[child[0], :], ord='fro')
            euclidean_distances1 = np.linalg.norm(agg_row1 - self.eigenvectors[child[0], :], ord='fro')
            if(euclidean_distances > euclidean_distances1):
                agg_row1 = np.expand_dims(agg_row1, axis=0)
                self.eigenvectors = np.vstack([self.eigenvectors, agg_row1])
                agg_row = agg_row1
            else:
                agg_row = np.expand_dims(agg_row, axis=0)
                self.eigenvectors = np.vstack([self.eigenvectors, agg_row])
            
            desired_ids = [child[0], child[1]]
            desired_layer = [entry['layer']
                             for entry in self.hierarchy if entry['id'] in desired_ids]
            desired_cluster_points = [entry['clusterPoints']
                                      for entry in self.hierarchy if entry['id'] in desired_ids]
            self.hierarchy.append({'eigenvectors': np.squeeze(agg_row, axis=0), 'id': i+self.n_cluster,
                                   'parents': (child[0], child[1]), 'layer': max(desired_layer) + 1,
                                   'clusterPoints': np.concatenate(desired_cluster_points, axis=0)})
            agg_cen = np.mean(
                self.centroids[[child[0], child[1]], :], axis=0).T
            self.centroids = np.vstack([self.centroids, agg_cen])

    def _get_hierarchy_components(self):
        leafs = list(range(self.n_cluster))
        children = self.model.children_
        count = self.n_cluster
        components = []
        components.append(leafs)
        for i, child in enumerate(children):
            comp = components[i]
            next = [x for x in comp if x not in child]
            next.append(count)
            components.append(next)
            count += 1
        return components

    def create_principle_components(self, *args):
        principal_components = []
        for i in args:
            for dic in self.hierarchy:
                if (dic['id'] == i):
                    principal_components.append(
                        (dic['clusterPoints'] - self.centroids[i]) @ dic['eigenvectors'])
        return principal_components

    def compute_reconstruction_error(self, *args):
        pcs = self.create_principle_components(*args)
        rec_data = []
        rec_error = []
        for index, pc in enumerate(pcs):
            rec_data.append(
                pc @ np.transpose(self.eigenvectors[args[index]]) + self.centroids[args[index]])

        cluster_points = [dic['clusterPoints']
                          for dic in self.hierarchy if dic['id'] in args]
        for i in range(len(args)):
            temp = np.sum(
                (rec_data[i] - cluster_points[i]) ** 2, axis=1).mean()
            rec_error.append(temp)
        return rec_error

    def calculate_mdl(self):
        # number of respected clusters in the model
        components_array = self._get_hierarchy_components()

        # fit the scalers
        rec_scaler = MinMaxScaler()
        min_rec = np.mean(self.compute_reconstruction_error(*components_array[0]))
        max_rec = np.mean(self.compute_reconstruction_error(*components_array[-1]))
        rec_array = np.asarray(
            [np.log2(min_rec), np.log2(max_rec)]).reshape(-1, 1)
        rec_scaler.fit(rec_array)

        param_scaler = MinMaxScaler()
        param_array = np.asarray(
            [np.log2(len(self.data)), self.n_cluster * np.log2(len(self.data))]).reshape(-1, 1)
        param_scaler.fit(param_array)

        mdl_array = []
        for elem in components_array:
            # mean rec. error of the selected clusters
            rec = np.mean(self.compute_reconstruction_error(*elem))
            num_params = len(elem)
            # MDL = Bits(model parameters) + Bits(data | model)
            mdl = 0.5 * param_scaler.transform(np.asarray(num_params * np.log2(len(
                self.data))).reshape(-1, 1)) + 0.5 * rec_scaler.transform(np.asarray(np.log2(rec)).reshape(-1, 1))
            mdl_array.append(mdl[0, 0])

        return mdl_array, components_array

    def mdl_info(self):
        mdl_array, components_array = self.calculate_mdl()
        blanks = '   '
        print('MDL for compression hierarchy\n----------------')
        for i, elem in enumerate(mdl_array):
            b = i * blanks
            print(f'{components_array[i]}{b}-->{elem}')
    
    def flatten_list(list):
        return [item for l in list for item in l]

    def search_best_cluster(data, n_cluster, n_dim):
        pcs = []
        mdl_scores = []
        subspaces = []
        min_mdl = float('inf') 
        model = None
        for i in range(1, n_dim + 1):
            while True:
                try:
                    model1 = myCluster(data, n_cluster, i)
                    break
                except:
                    pass
            mdl, comps = model1.calculate_mdl()
            mdl_scores.append(mdl)
            subspaces.append([len(j) for j in comps])
            pcs.append([i for _ in comps])
            min_val = min(mdl)
            if min_val < min_mdl:
                model = model1
                min_mdl = min_val

        # Initialize figure for both plots
        fig = plt.figure(figsize=(12, 6))

        # 3D landscape plot
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        k = [item for sublist in subspaces for item in sublist]
        l = [item for sublist in pcs for item in sublist]
        rl = [item for sublist in mdl_scores for item in sublist]
        triang = mtri.Triangulation(k, l)
        ax1.plot_trisurf(triang, rl, cmap='jet')
        ax1.scatter(k, l, rl, marker='.', s=10, c="black", alpha=0.5)
        ax1.view_init(elev=50, azim=-45)
        ax1.set_xticks(np.arange(1, max(k) + 1))
        ax1.set_yticks(np.arange(1, n_dim + 1))
        ax1.set_xlabel('# Subspaces')
        ax1.set_ylabel('# Principal Components')
        ax1.set_zlabel('MDL Score')

        # 2D plot
        ax2 = fig.add_subplot(1, 2, 2)
        for i in range(len(subspaces)):
            ax2.plot(subspaces[i], mdl_scores[i], label=f'{i+1} PC')
        ax2.set_xlabel('# Subspaces')
        ax2.set_ylabel('MDL Score')
        ax2.legend()

        # Adjust layout
        plt.subplots_adjust(wspace=0.4)

        plt.show()

        return model
    
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

    def plot_mdl_stats(self):
        mdl_array, components_array = self.calculate_mdl()
        num_subspaces = [len(i) for i in components_array]
        plt.plot(num_subspaces, mdl_array)
        plt.xlabel('# Subspaces')
        plt.ylabel('MDL Score')

    def plot_elbow(self, n_cluster):
        all_eigenvalues = []
        components_array = self._get_hierarchy_components()
        components_array = [elem for elem in components_array if len(elem) == n_cluster]
        for i in components_array[0]:
            c_points = [dic['clusterPoints'] - self.centroids[i] for dic in self.hierarchy if dic['id'] == i]
            eigenvalues , _ = np.linalg.eig(np.transpose(c_points[0]) @ c_points[0])
            all_eigenvalues.append(eigenvalues)
        return all_eigenvalues
            
    def plot_elbow_plots_with_cutoff(self, n_cluster, cols):
        all_eigenvalues = self.plot_elbow(n_cluster)
        # Determine the number of plots based on the number of elements in all_eigenvalues
        n_plots = len(all_eigenvalues)
        
        # Calculate the number of rows and columns for the subplots
        n_cols = cols  # Adjust this based on your preference or display size
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ensures there are enough rows
        
        # Create a figure with subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        
        # Flatten the array of axes for easy iteration
        axs = axs.flatten()
        
        for i, eigenvalues in enumerate(all_eigenvalues):
            # Sort the eigenvalues in descending order
            sorted_eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Calculate the cumulative sum of the eigenvalues
            cumsum_eigenvalues = np.cumsum(sorted_eigenvalues)
            total = cumsum_eigenvalues[-1]
            
            # Find the number of components required to reach 80% of total variance
            cutoff_index = np.argmax(cumsum_eigenvalues >= total * 0.8) + 1  # +1 for the position in a 1-indexed sense
            
            # Plot the eigenvalues
            axs[i].plot(sorted_eigenvalues, '-o', markersize=4, label='Eigenvalues')
            axs[i].set_title(f'Cluster {i+1}')
            axs[i].set_xlabel('Index')
            axs[i].set_ylabel('Eigenvalue')
            
            # Add a vertical line at the cutoff index
            axs[i].axvline(x=cutoff_index, color='r', linestyle='--', label=f'80% cutoff at {cutoff_index}')
            axs[i].legend()
            axs[i].grid(True)
        
        # Hide any unused subplot areas if the number of plots is not a perfect fit for the grid
        for ax in axs[n_plots:]:
            ax.axis('off')
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()

    def plot_2d_scatter(self, dim1, dim2, *args, vectors=False):
        n_categories = len(args)
        # Get the Viridis colormap
        viridis_cmap = plt.cm.get_cmap('viridis', n_categories)
        # Generate a list of colors
        colors = [viridis_cmap(i) for i in range(n_categories)]
        # Access the VQPCA clustering solution:
        for index, i in enumerate(args):
            cluster_points = [dic['clusterPoints']
                              for dic in self.hierarchy if dic['id'] == i]
            cluster_points = np.squeeze(cluster_points, axis=0)
            plt.scatter(cluster_points[:, dim1], cluster_points[:, dim2],
                        color=colors[index], label=f'Cluster {i} ({len(cluster_points)})')

        if(vectors):
            for i in range(self.n_components):
                plt.quiver(self.centroids[args, dim1], self.centroids[args, dim2],
                        self.eigenvectors[args, dim1, i], self.eigenvectors[args, dim2, i],
                        color=colors, scale=5, edgecolor='black', linewidth=1)

        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        # Show the
        # Show legend
        plt.legend()
        plt.show()

    def info(self):
        for index, dic in enumerate(self.hierarchy):
            id = dic['id']
            parents = dic['parents']
            rec_error = self.compute_reconstruction_error(id)
            num_points = len(dic['clusterPoints'])
            if (index != 0):
                print('------------------------------')
            print(f'Cluster_{id}:\n-> Number of Points: {num_points}\n-> Parents: {parents}\n-> Rec. Error: {rec_error}')
