##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import scipy
from scipy.spatial import distance
from scipy import special
import math
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import util.util as util
import copy
from tqdm import tqdm
from typing import List, Tuple


class DistributionBasedOutlierDetection:
    """
    Class for outlier detection algorithms based on some distribution of the data. They all consider only single
    points per row (i.e. one column).
    """

    @staticmethod
    def chauvenet(data_table: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Find outliers in the specified column of datatable based on the Chauvenet-Criterion (assuming the data is
        normally distributed and calculating the probability that a data point belongs to the distribution) and add a
        binary column with the same name extended with '_outlier' that expresses the result per data point. As
        criterion 1/(2*N) is used, where N is the number of points.
        Taken partly from: https://www.astro.rug.nl/software/kapteyn/

        :param data_table: DataFrame with data the outlier detection will be applied on.
        :param col: Name of the column to process.
        :return: Original DataFrame with a new binary column added.
        """

        # Compute the mean and standard deviation
        mean = data_table[col].mean()
        std = data_table[col].std()
        N = len(data_table.index)
        criterion = 1.0 / (2 * N)

        # Consider the deviation for the data points
        deviation = abs(data_table[col] - mean) / std

        # Express the upper and lower bounds
        low = -deviation / math.sqrt(2)
        high = deviation / math.sqrt(2)
        prob = []
        mask = []

        # Iterate over all rows in the data
        for i in range(0, len(data_table.index)):
            # Determine the probability of observing the point
            prob.append(1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i])))
            # Mark as an outlier when the probability is below our criterion
            mask.append(prob[i] < criterion)
        data_table[col + '_outlier'] = mask
        return data_table

    @staticmethod
    def mixture_model(data_table: pd.DataFrame, col: str, components: int = 3) -> pd.DataFrame:
        """
        Fit a gaussian mixture model towards the data expressed in col and adds a column with the probability of
        observing the value given the mixture model. New column name is the col name extended with '_mixture'.

        :param data_table: DataFrame containing the data the mixture model is applied on.
        :param col: Name of the column that is processed.
        :param components: Number of normal distributions to use when fitting the mixture model.
        :return: Original data with column added, that contains the predicted probabilities.
        """

        # Fit a mixture model to our data.
        data = data_table[data_table[col].notnull()][col]
        mixture_model = GaussianMixture(n_components=components, max_iter=100, n_init=1)
        reshaped_data = np.array(data.values.reshape(-1, 1))
        mixture_model.fit(reshaped_data)

        # Predict the probabilities
        probabilities = mixture_model.score_samples(reshaped_data)
        # Create a data frame and concatenate with the original data
        df_probabilities = pd.DataFrame(np.power(10, probabilities), index=data.index, columns=[col + '_mixture'])
        data_table = pd.concat([data_table, df_probabilities], axis=1)
        return data_table


class DistanceBasedOutlierDetection:
    """
    Class providing methods for distance based outlier detection.
    """

    def __init__(self):
        self.distances = None

    @staticmethod
    def create_distance_table(data_table: pd.DataFrame, cols: List[str], d_function: str) -> pd.DataFrame:
        """
        Create distance table between rows in the data table. Only cols are considered and the specified  distance
        function is used to compute the distance .

        :param data_table: DataFrame to calculate distance matrix for.
        :param cols: Cols to use for calculating distance between rows.
        :param d_function: Distance function to use for calculation. By now only euclidean is supported.
        :return: NxN matrix with distance from each point to each where N is the number of rows.
        """

        data_table[cols] = data_table.loc[:, cols].astype('float32')
        print('Calculating distance matrix, this may take a while and your computer may be slower.')
        return pd.DataFrame(scipy.spatial.distance.squareform(util.distance(data_table.loc[:, cols], d_function)),
                            columns=data_table.index, index=data_table.index).astype('float32')

    def simple_distance_based(self, data_table: pd.DataFrame, cols: List[str], d_function: str, d_min: float,
                              f_min: float) -> pd.DataFrame:
        """
        Detect outliers using a simple distance based algorithm. Assuming a distance function, e.g. 'euclidean',
        a minimum distance of neighboring points and frequency of occurrence, outliers are detected and a new binary
        column is added.

        :param data_table: Data to detect outliers in.
        :param cols: Columns to use for calculating distance between rows.
        :param d_function: Distance function to use for calculating the distance between points.
        :param d_min: Minimum distance to count points as neigbours.
        :param f_min: Proportion of all data points from which a point is counted as an outlier.
        :return: Original data with new binary column names 'simple_dist_outlier'.
        """

        print('Calculating simple distance-based criterion.')

        # Normalize the dataset first
        norm_data_table = util.normalize_dataset(data_table.dropna(axis=0, subset=cols), cols)
        # Create the distance table first between all instances
        distances = self.create_distance_table(norm_data_table, cols, d_function)

        mask = []
        # Pass the rows in our table
        for i in tqdm(range(0, len(norm_data_table.index))):
            # Check what faction of neighbors are beyond dmin
            frac = (float(sum([1 for col_val in distances.iloc[i, :].tolist() if col_val > d_min])) / len(
                norm_data_table.index))
            # Mark as an outlier if beyond the minimum frequency
            mask.append(frac > f_min)
        data_mask = pd.DataFrame(mask, index=norm_data_table.index, columns=['simple_dist_outlier'])
        data_table = pd.concat([data_table, data_mask], axis=1)
        return data_table

    def local_outlier_factor(self, data_table: pd.DataFrame, cols: List[str], d_function: str, k: int) -> pd.DataFrame:
        """
        Compute the local outlier factor for each row in the data table. Inspired by
        https://github.com/damjankuznar/pylof/blob/master/lof.py but tailored towards the distance # metrics and data
        structures used here.

        :param data_table: DataFrame to calculate the lof for.
        :param cols: Cols to use for calculating the distance between rows.
        :param d_function: Distance function to use. By now only 'euclidean' is supported.
        :param k: Number of neighboring points considered.
        :return: Original DataFrame with new column named 'lof' added.
        """

        print("Calculating local outlier factor.")

        # Normalize the dataset first.
        norm_data_table = util.normalize_dataset(data_table.dropna(axis=0, subset=cols), cols)
        # Create the distance table first between all instances:
        self.distances = self.create_distance_table(norm_data_table, cols, d_function)

        outlier_factor = []
        # Compute the outlier score per row.
        for i in tqdm(range(0, len(norm_data_table.index))):
            outlier_factor.append(self.local_outlier_factor_instance(i, k))
        data_outlier_probs = pd.DataFrame(outlier_factor, index=norm_data_table.index, columns=['lof'])
        data_table = pd.concat([data_table, data_outlier_probs], axis=1)
        del self.distances
        return data_table

    def reachability_distance(self, k: int, i1: int, i2: int) -> float:
        """
        Calculate the distance between a row i1 and i2.

        :param k: Number of neighbours to use.
        :param i1: Index of row one.
        :param i2: Index of row two.
        :return: Maximum of the k-distance of i2 and the real distance between i1 and i2.
        """

        # Compute the k-distance of i2
        k_distance_value, neighbors = self.k_distance(i2, k)
        # The value is the max of the k-distance of i2 and the real distance
        return max([k_distance_value, self.distances.iloc[i1, i2]])

    def local_reachability_density(self, i: int, k: int, k_distance_i, neighbors_i) -> float:
        """
        Compute the local reachability density for a row i, given a k-distance and set of neighbors.

        :param i: Index of row i.
        :param k: Number of neighbours.
        :param k_distance_i: Maximum distance within k neighbours of row i.
        :param neighbors_i: Indicies of the neighbours of row i.
        :return: Number of neighbors divided by the sum of the reachability distances.
        """

        # Set distances to neighbors to 0
        reachability_distances_array = [0.0] * len(neighbors_i)

        # Compute the reachability distance between i and all neighbors
        for i, neighbor in enumerate(neighbors_i):
            reachability_distances_array[i] = self.reachability_distance(k, i, neighbor)
        if not any(reachability_distances_array):
            return float("inf")
        else:
            # Return the number of neighbors divided by the sum of the reachability distances
            return len(neighbors_i) / sum(reachability_distances_array)

    def k_distance(self, i: int, k: int) -> Tuple[float, List[int]]:
        """
        Compute the k-distance of a row i, namely the maximum distance within the k nearest neighbors and return a
        tuple containing this value and the neighbors within this distance.

        :param i: Index of row to find the nearest k neighbours for.
        :param k: Number of neighbours to find.
        :return: Maximum distance within k neighbours and indexes of k nearest neighbours.
        """

        # Look up the values in the distance table, select the min_pts^th lowest value and take the value pairs
        # Take min_pts + 1 as the instance itself is in there
        neighbors = np.argpartition(np.array(self.distances.iloc[i, :]), k + 1)[0:(k + 1)].tolist()
        if i in neighbors:
            neighbors.remove(i)
        return max(self.distances.iloc[i, neighbors]), neighbors

    def local_outlier_factor_instance(self, i: int, k: int) -> float:
        """
        Compute the local outlier score of row i given a setting for k nearest neighbours.

        :param i: Index of row to calculate lof for.
        :param k: Number of neighbours to consider.
        :return: Average ratio between local reachibility and reachibility of neighbours.
        """

        # Compute the k-distance for i
        k_distance_value, neighbors = self.k_distance(i, k)
        # Compute the local reachability given the found k-distance and neighbors
        instance_lrd = self.local_reachability_density(i, k, k_distance_value, neighbors)
        lrd_ratios_array = [0.0] * len(neighbors)

        # Compute the k-distances and local reachability density of the neighbors
        for i, neighbor in enumerate(neighbors):
            k_distance_value_neighbor, neighbors_neighbor = self.k_distance(neighbor, k)
            neighbor_lrd = self.local_reachability_density(neighbor, k, k_distance_value_neighbor, neighbors_neighbor)
            # Store the ratio between the neighbor and the row i
            lrd_ratios_array[i] = neighbor_lrd / instance_lrd

        # Return the average ratio
        return sum(lrd_ratios_array) / len(neighbors)
