##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 5                                               #
#                                                            #
##############################################################

from sklearn.cluster import KMeans
from Chapter5.DistanceMetrics import InstanceDistanceMetrics
import sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from Chapter5.DistanceMetrics import PersonDistanceMetricsNoOrdering
from Chapter5.DistanceMetrics import PersonDistanceMetricsOrdering
import random
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import DistanceMetric
import pyclust
from typing import Tuple, List


class NonHierarchicalClustering:
    """
    Implementation of non hierarchical clustering approaches.
    """

    # Global parameters for distance functions
    def __init__(self, p: int = 1, max_lag: int = 1):
        self.ranges = []
        self.p = p
        self.max_lag = max_lag

    # Identifiers of the various distance and abstraction approaches
    euclidean = 'euclidean'
    minkowski = 'minkowski'
    manhattan = 'manhattan'
    gower = 'gower'
    abstraction_mean = 'abstraction_mean'
    abstraction_normal = 'abstraction_normal'
    abstraction_p = 'abstraction_p'
    abstraction_euclidean = 'abstract_euclidean'
    abstraction_lag = 'abstract_lag'
    abstraction_dtw = 'abstract_dtw'

    def gowers_similarity(self, X: pd.DataFrame, Y: pd.DataFrame) -> np.ndarray:
        """
        Define the gowers distance between arrays to be used in k-means and k-medoids

        :param X: First DataFrame to calculate gowers similarity for.
        :param Y: Second DataFrame to pair up with X.
        :return: Distances between all paired rows in X and Y.
        """

        # Convert input values to numpy arrays
        X = np.matrix(X)
        Y = np.matrix(Y)
        distances = np.zeros(shape=(X.shape[0], Y.shape[0]))
        DM = InstanceDistanceMetrics()
        # Pair up the elements in the dataset
        for x_row in range(0, X.shape[0]):
            data_row1 = pd.DataFrame(X[x_row])
            for y_row in range(0, Y.shape[0]):
                data_row2 = pd.DataFrame(Y[y_row]).transpose()
                # Compute the distance as defined in distance metrics class
                distances[x_row, y_row] = DM.gowers_similarity(data_row1, data_row2, self.p)
        return np.array(distances)

    def minkowski_distance(self, X, Y=None) -> np.ndarray:
        """
        Calculate the Minkowski distance using a predefined distance function.

        :param X: First matrix like component.
        :param Y: Optional second matrix like component.
        :return: Distance matrix as Numpy array.
        """

        dist = DistanceMetric.get_metric('minkowski', p=self.p)
        return dist.pairwise(X, Y)

    @staticmethod
    def manhattan_distance(X, Y=None) -> np.ndarray:
        """
        Calculate the Manhattan distance using a predefined distance function.

        :param X: First matrix like component.
        :param Y: Optional second matrix like component.
        :return: Distance matrix as Numpy array.
        """

        dist = DistanceMetric.get_metric('manhattan')
        return dist.pairwise(X, Y)

    # Use a predefined distance function for the Euclidean distance
    @staticmethod
    def euclidean_distance(X, Y=None) -> np.ndarray:
        """
        Calculate the Euclidean distance using a predefined distance function.

        :param X: First matrix like component.
        :param Y: Optional second matrix like component.
        :return: Distance matrix as Numpy array.
        """

        dist = DistanceMetric.get_metric('euclidean')
        return dist.pairwise(X, Y)

    def aggregate_datasets(self, datasets: List[pd.DataFrame], cols: List[str], abstraction_method: str) \
            -> pd.DataFrame:
        """
        Flatten each dataset to a single record/instance for comparing datasets between persons. This is done based on
        the approaches defined in the distance metrics file.

        :param datasets: List of DataFrames to aggregate.
        :param cols: Columns to keep while aggregating.
        :param abstraction_method: Abstraction method to use for aggregation.
        :return: Aggregated DataFrame.
        """

        temp_datasets = []
        DM = PersonDistanceMetricsNoOrdering()

        # Flatten all datasets and add them to the newly formed dataset
        for i in range(0, len(datasets)):
            temp_dataset = datasets[i][cols]
            temp_datasets.append(temp_dataset)

        if abstraction_method == self.abstraction_normal:
            return DM.create_instances_normal_distribution(temp_datasets)
        else:
            return DM.create_instances_mean(temp_datasets)

    def k_means_over_instances(self, dataset: pd.DataFrame, cols: List[str], k: int, distance_metric: str,
                               max_iters: int, n_inits: int, p: int = 1) -> pd.DataFrame:
        """
        Perform k-means over an individual dataset.

        :param dataset: DataFrame to apply clustering on.
        :param cols: Columns to use for clustering.
        :param k: Number of clusters.
        :param distance_metric: Distance metric to use for clustering.
        :param max_iters: Maximum number of iterations.
        :param n_inits: Number of inits.
        :param p: Parameter p in case of Minkowski distance.
        :return: Original DataFrame
        """

        # Take the appropriate columns
        temp_dataset = dataset[cols]
        # Override the standard distance functions. Store the original first
        sklearn_euclidian_distances = sklearn.metrics.pairwise.euclidean_distances
        if distance_metric == self.euclidean:
            sklearn.metrics.pairwise.euclidean_distances = self.euclidean_distance
        elif distance_metric == self.minkowski:
            self.p = p
            sklearn.metrics.pairwise.euclidean_distances = self.minkowski_distance
        elif distance_metric == self.manhattan:
            sklearn.metrics.pairwise.euclidean_distances = self.manhattan_distance
        elif distance_metric == self.gower:
            for col in temp_dataset.columns:
                self.ranges.append(temp_dataset[col].max() - temp_dataset[col].min())
            sklearn.metrics.pairwise.euclidean_distances = self.gowers_similarity
        # If the set distance function is unknown, the default euclidean distance function is used
        # Apply the k-means algorithm
        kmeans = KMeans(n_clusters=k, max_iter=max_iters, n_init=n_inits, random_state=0).fit(temp_dataset)
        # Add the labels to the dataset
        dataset['cluster'] = kmeans.labels_
        # Compute the silhouette and add it as well
        silhouette_per_inst = silhouette_samples(temp_dataset, kmeans.labels_)
        dataset['silhouette'] = silhouette_per_inst

        # Reset the module distance function for further usage
        sklearn.metrics.pairwise.euclidean_distances = sklearn_euclidian_distances

        return dataset

    def k_means_over_datasets(self, datasets: List[pd.DataFrame], cols: List[str], k: int, abstraction_method: str,
                              distance_metric: str, max_iters: int, n_inits: int, p: int = 1) -> pd.DataFrame:
        """
        Perform k-means clustering over multiple datasets by flattening them into a single instance using the set
        abstraction method. Them the simple k-means over dataset method is applied.

        :param datasets: List of DataFrames.
        :param cols: Columns to use for clustering.
        :param k: Number of clusters.
        :param abstraction_method: Aggregation method for flattening the datasets.
        :param distance_metric: Distance function to use for clustering.
        :param max_iters: Maximum number of iterations.
        :param n_inits: Number of inits.
        :param p: Optional parameter p for Minkowski distance metrics.
        :return: Original DataFrame with cluster and silhouette score columns added.
        """

        # Flatten the datasets into one instance
        temp_dataset = self.aggregate_datasets(datasets, cols, abstraction_method)

        # Apply the instance based algorithm
        return self.k_means_over_instances(temp_dataset, temp_dataset.columns, k, distance_metric, max_iters, n_inits,
                                           p)

    def compute_distance_matrix_instances(self, dataset: pd.DataFrame, distance_metric: str) -> pd.DataFrame:
        """
        Compute a complete distance matrix between rows / point in dataset. For k-medoids algorithm the implemented
        algorithm is used.

        :param dataset: Pandas DataFrame to calculate distance matrix for.
        :param distance_metric: Distance function to use.
        :return: Distances between all rows in form of DataFrame.
        """

        # If the distance function is not defined in distance metrics, use the standard euclidean distance
        if not (distance_metric in [self.manhattan, self.minkowski, self.gower, self.euclidean]):
            distances = sklearn.metrics.pairwise.euclidean_distances(X=dataset, Y=dataset)
            return pd.DataFrame(distances, index=range(0, len(dataset.index)), columns=range(0, len(dataset.index)))
        # Create an empty pandas dataframe for our distance matrix
        distances = pd.DataFrame(index=range(0, len(dataset.index)), columns=range(0, len(dataset.index)))

        # Define the ranges of the columns if we use the gowers distance
        if distance_metric == self.gower:
            for col in dataset.columns:
                self.ranges.append(dataset[col].max() - dataset[col].min())

        # Compute the distances for each pair. Note that the distances are assumed to be symmetric
        for i in range(0, len(dataset.index)):
            for j in range(i, len(dataset.index)):
                if distance_metric == self.manhattan:
                    distances.iloc[i, j] = self.manhattan_distance(dataset.iloc[i:i + 1, :], dataset.iloc[j:j + 1, :])
                elif distance_metric == self.minkowski:
                    distances.iloc[i, j] = self.manhattan_distance(dataset.iloc[i:i + 1, :], dataset.iloc[j:j + 1, :])
                elif distance_metric == self.gower:
                    distances.iloc[i, j] = self.gowers_similarity(dataset.iloc[i:i + 1, :], dataset.iloc[j:j + 1, :])
                elif distance_metric == self.euclidean:
                    distances.iloc[i, j] = self.euclidean_distance(dataset.iloc[i:i + 1, :], dataset.iloc[j:j + 1, :])
                distances.iloc[j, i] = distances.iloc[i, j]
        return distances

    def k_medoids_over_instances(self, dataset: pd.DataFrame, cols: List[str], k: int, distance_metric: str,
                                 max_iters: int, n_inits: int = 5, p: int = 1):
        """
        Apply k-medoids clustering using the self implemented distance metrics.

        :param dataset: DataFrame to apply clustering on.
        :param cols: List of columns to use for clustering.
        :param k: Number of clusters.
        :param distance_metric: Distance metrics to use for clustering.
        :param max_iters: Maximum number of iterations.
        :param n_inits: Number of inits.
        :param p: Optional parameter for Minkowski distance metrics.
        :return: Original DataFrame with cluster and silhouette score columns added.
        """

        # Select the appropriate columns
        temp_dataset = dataset[cols]
        # Use PyClust Package in case of default distance metric
        if distance_metric == 'default':
            km = pyclust.KMedoids(n_clusters=k, n_trials=n_inits)
            km.fit(temp_dataset.values)
            cluster_assignment = km.labels_
        else:
            self.p = p
            cluster_assignment = []
            best_silhouette = -1

            # Compute all distances
            D = self.compute_distance_matrix_instances(temp_dataset, distance_metric)

            for it in range(0, n_inits):
                # Select k random points as centers first
                centers = random.sample(range(0, len(dataset.index)), k)
                prev_centers = []

                n_iter = 0
                while (n_iter < max_iters) and not (centers == prev_centers):
                    n_iter += 1
                    prev_centers = centers
                    # Assign points to clusters
                    points_to_centroid = D[centers].idxmin(axis=1)

                    new_centers = []
                    for i in range(0, k):
                        # Find the new center that minimized the sum of the differences

                        best_center = D.loc[points_to_centroid == centers[i]].sum().idxmin(axis=1)
                        new_centers.append(best_center)
                    centers = new_centers

                # Convert centroids to cluster numbers:
                points_to_centroid = D[centers].idxmin(axis=1)
                current_cluster_assignment = []
                for i in range(0, len(dataset.index)):
                    current_cluster_assignment.append(centers.index(points_to_centroid.iloc[i]))

                silhouette_avg = silhouette_score(temp_dataset, np.array(current_cluster_assignment))
                if silhouette_avg > best_silhouette:
                    cluster_assignment = current_cluster_assignment
                    best_silhouette = silhouette_avg

        # Add the clusters and silhouette scores to the dataset
        dataset['cluster'] = cluster_assignment
        silhouette_per_inst = silhouette_samples(temp_dataset, np.array(cluster_assignment))
        dataset['silhouette'] = silhouette_per_inst

        return dataset

    def compute_distance_matrix_datasets(self, datasets: List[pd.DataFrame], distance_metric: str) -> pd.DataFrame:
        """
        Compute the pairwise distance matrix for a list of datasets. The distance matrix is used for the implementation
        of k-medoids.

        :param datasets: List of DataFrames.
        :param distance_metric: Distance metric to use for calculating distance between datasets.
        :return: Pairwise distance matrix.
        """

        distances = pd.DataFrame(index=range(0, len(datasets)), columns=range(0, len(datasets)))
        DMNoOrdering = PersonDistanceMetricsNoOrdering()
        DMOrdering = PersonDistanceMetricsOrdering()

        # Compute the distances for each pair. Distanced are assumed to be symmetric
        for i in range(0, len(datasets)):
            for j in range(i, len(datasets)):
                if distance_metric == self.abstraction_p:
                    distances.iloc[i, j] = DMNoOrdering.p_distance(datasets[i], datasets[j])
                elif distance_metric == self.abstraction_euclidean:
                    distances.iloc[i, j] = DMOrdering.euclidean_distance(datasets[i], datasets[j])
                elif distance_metric == self.abstraction_lag:
                    distances.iloc[i, j] = DMOrdering.lag_correlation(datasets[i], datasets[j], self.max_lag)
                elif distance_metric == self.abstraction_dtw:
                    distances.iloc[i, j] = DMOrdering.dynamic_time_warping(datasets[i], datasets[j])
                distances.iloc[j, i] = distances.iloc[i, j]
        return distances

    # Note: distance metric only important in combination with certain abstraction methods as we allow for more
    # in k-medoids.
    def k_medoids_over_datasets(self, datasets: List[pd.DataFrame], cols: List[str], k: int, abstraction_method: str,
                                distance_metric: str, max_iters: int, n_inits: int = 5, p: int = 1, max_lag: int = 5)\
            -> pd.DataFrame:
        """
        Apply k-medoids clustering on multiple datasets by flattening them into a single instance using the given
        abstraction method.

        :param datasets: List of DataFrames to apply clustering on.
        :param cols: List of columns to use for clustering.
        :param k: Number of clusters.
        :param abstraction_method: Aggregation method for flattening the datasets.
        :param distance_metric: Distance metrics to use for clustering.
        :param max_iters: Maximum number of iterations.
        :param n_inits: Number of inits.
        :param p: Optional parameter for Minkowski distance metric.
        :param max_lag: Maximum lag.
        :return: Original DataFrame with cluster column added.
        """

        self.p = p
        self.max_lag = max_lag

        # Flatten the datasets to be able to apply simple k-medoids clustering
        if abstraction_method in [self.abstraction_mean, self.abstraction_normal]:
            # Convert the datasets to instances
            temp_dataset = self.aggregate_datasets(datasets, cols, abstraction_method)

            # Apply the instance based algorithm in case of known abstraction method
            return self.k_medoids_over_instances(temp_dataset, temp_dataset.columns, k, distance_metric, max_iters,
                                                 n_inits=n_inits, p=p)

        # For the case over datasets there is no quality metric, therefore just look at a single initialization
        # Select k random points as centers first
        centers = random.sample(range(0, len(datasets)), k)
        prev_centers = []
        # Compute all distances
        D = self.compute_distance_matrix_datasets(datasets, abstraction_method)

        n_iter = 0
        while (n_iter < max_iters) and not (centers == prev_centers):
            n_iter += 1
            prev_centers = centers
            # Assign points to clusters
            points_to_centroid = D[centers].idxmin(axis=1)

            new_centers = []
            for i in range(0, k):
                # Find the new center that minimized the sum of the differences
                best_center = D.loc[points_to_centroid == centers[i], points_to_centroid == centers[i]].sum().idxmin(
                    axis=1)
                new_centers.append(best_center)
            centers = new_centers

        # Convert centroids to cluster numbers
        points_to_centroid = D[centers].idxmin(axis=1)
        cluster_assignment = []
        for i in range(0, len(datasets)):
            cluster_assignment.append(centers.index(points_to_centroid.iloc[i, :]))

        dataset = pd.DataFrame(index=range(0, len(datasets)))
        dataset['cluster'] = cluster_assignment

        # Silhouette cannot be used here as it used a distance between instances, not datasets
        return dataset


class HierarchicalClustering:
    """
    This class implements all distances except the Gowers distance between instances.
    Furthermore, only the agglomerative approach is implemented.
    """

    def __init__(self):
        self.link = None

    def agglomerative_over_instances(self, dataset: pd.DataFrame, cols: List[str], max_clusters: int,
                                     distance_metric: str, use_prev_linkage: bool = False,
                                     link_function: str = 'single') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Perform agglomerative clustering over a single dataset using the given cols. The function uses the specified
        distance metric via scipy linkage function.

        :param dataset: Pandas DataFrame with the data to perform clustering on.
        :param cols: List of columns to use for clustering.
        :param max_clusters: Number of maximum clusters to use.
        :param distance_metric: Distance metric to use during clustering.
        :param use_prev_linkage: Set to True if the previous linkage function should be used.
        :param link_function: Parameter for scipy linkage function.
        :return: The original DataFrame with cluster and silhouette columns added and the linkage.
        """

        # Select the relevant columns
        temp_dataset = dataset[cols]
        df = NonHierarchicalClustering()

        # Perform the clustering process according to the specified distance metric
        if (not use_prev_linkage) or (self.link is None):
            if distance_metric == df.manhattan:
                self.link = linkage(temp_dataset.values, method=link_function, metric='cityblock')
            else:
                self.link = linkage(temp_dataset.values, method=link_function, metric='euclidean')

        # Assign the clusters given the set maximum and compute the silhouette score
        cluster_assignment = fcluster(self.link, max_clusters, criterion='maxclust')
        dataset['cluster'] = cluster_assignment
        silhouette_per_inst = silhouette_samples(temp_dataset, np.array(cluster_assignment))
        dataset['silhouette'] = silhouette_per_inst

        return dataset, self.link

    def agglomerative_over_datasets(self, datasets: List[pd.DataFrame], cols: List[str], max_clusters: int,
                                    abstraction_method: str, distance_metric: str, use_prev_linkage: bool = False,
                                    link_function: str = 'single') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Perform agglomerative clustering over the datasets by flattening them into a single dataset.

        :param datasets: List of DataFrames.
        :param cols: List of columns to use.
        :param max_clusters: Number of maximum clusters to use.
        :param abstraction_method: Method to use when flattening the datasets into one instance.
        :param distance_metric: Distance metric to use during clustering.
        :param use_prev_linkage: Set to True if the previous linkage function should be used.
        :param link_function: Parameter for scipy linkage function.
        :return: The original DataFrame with cluster and silhouette columns added and the linkage.
        """

        # Convert the datasets to instances
        df = NonHierarchicalClustering()
        temp_dataset = df.aggregate_datasets(datasets, cols, abstraction_method)

        # Apply the instance based algorithm
        return self.agglomerative_over_instances(temp_dataset, temp_dataset.columns, max_clusters, distance_metric,
                                                 use_prev_linkage=use_prev_linkage, link_function=link_function)
