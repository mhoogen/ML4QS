##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 5                                               #
#                                                            #
##############################################################

import numbers
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import stats
import sys
from sklearn.neighbors import DistanceMetric
import sklearn
from typing import List, Union


class InstanceDistanceMetrics:
    """
    Class defining the distance metrics that are not available as standard ones.
    """

    # S for gowers distance
    @staticmethod
    def s(val1, val2, num_range: float) -> Union[int, float]:
        """
        Calculate S for gowers distance. In case of numerical values the normalized absolute difference is returned,
        otherwise 1 if values are equal.

        :param val1: First value for comparison.
        :param val2: Second value for comparison.
        :param num_range: Range in case of numerical values.
        :return: Normalized absolute difference for numerical values / 1 in case of equal values, otherwise 0.
        """

        # If both values are numbers take the difference and normalize
        if isinstance(val1, numbers.Real) and isinstance(val1, numbers.Real):
            return 1 - (float(abs(val1 - val2)) / num_range)
        # If comparing something else, we just look at whether the values are equal
        else:
            if val1 == val2:
                return 1
            else:
                return 0

    @staticmethod
    def delta(val1, val2):
        """
        Calculate the delta between two values for gowers distance.

        :param val1: First value.
        :param val2: Second value.
        :return: 1 if both values are not NaN, otherwise 0.
        """
        # Check whether both values are known (i.e. nan), if so the delta is 1, 0 otherwise
        if (not np.isnan(val1)) and (not np.isnan(val2)):
            return 1
        return 0

    def gowers_similarity(self, data_row1: pd.DataFrame, data_row2: pd.DataFrame, ranges: List[float]) -> float:
        """
        Define gowers distance between two rows, given the ranges of the numerical variables over the entire dataset
        (over all columns in row1 and row2).

        :param data_row1: First row for gowers similarity.
        :param data_row2: Second row for gowers similarity.
        :param ranges: List of the ranges for the numerical variables. Must have the same length as columns in rows.
        :return: Returns the gowers similarity. 0 means rows are identical, 1 means rows are maximum different.
        """

        # Similarity cannot be computed if rows do not have the same length
        if len(data_row1.columns) != len(data_row2.columns):
            return -1

        delta_total = 0
        s_total = 0

        # Iterate over all columns
        for i in range(0, len(data_row1.columns)):
            val1 = data_row1[data_row1.columns[i]].values[0]
            val2 = data_row2[data_row2.columns[i]].values[0]
            # Compute the delta
            delta = self.delta(val1, val2)
            delta_total += delta
            # Compute s if the delta is above 0
            if delta > 0:
                s_total += self.s(val1, val2, ranges[i])
        return float(s_total) / delta_total


class PersonDistanceMetricsNoOrdering:
    """
    Class to flatten datasets or compute the statistical difference between cases.
    """

    gower = 'gower'
    minkowski = 'minkowski'

    @staticmethod
    def create_instances_mean(datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate datasets and return a new dataset with aggregated data instances based on the mean values in the rows.
        All datasets must have the same columns.

        :param datasets: List of DataFrames.
        :return: DataFrame with means of all rows from each given DataFrame.
        """

        index = range(0, len(datasets))
        cols = datasets[0].columns
        new_dataset = pd.DataFrame(index=index, columns=cols)

        for i in range(0, len(datasets)):
            for col in cols:
                # Compute the mean per column and assign that value for the row representing the current dataset
                new_dataset.iloc[i, new_dataset.columns.get_loc(col)] = datasets[i][col].mean()

        return new_dataset

    #
    @staticmethod
    def create_instances_normal_distribution(datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate datasets by fitting their cols to normal distribution and use parameters as instances. All datasets
        must have the same columns.

        :param datasets:
        :return: DataFrame with normal distribution parameters of each col for each dataset.
        """

        index = range(0, len(datasets))
        cols = datasets[0].columns
        new_cols = []
        # Create new columns for the parameters of the distribution
        for col in cols:
            new_cols.append(f'{col}_mu')
            new_cols.append(f'{col}_sigma')
        new_dataset = pd.DataFrame(index=index, columns=new_cols)

        for i in range(0, len(datasets)):
            for col in cols:
                # Fit the distribution and assign the values to the row representing the dataset
                mu, sigma = norm.fit(datasets[i][col])
                new_dataset.iloc[i, new_dataset.columns.get_loc(col + '_mu')] = mu
                new_dataset.iloc[i, new_dataset.columns.get_loc(col + '_sigma')] = sigma

        return new_dataset

    @staticmethod
    def p_distance(dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> float:
        """
        Calculate the distance between datasets based on the Kolmogorov-Smirnov statistic. Both datasets must have the
        same columns.

        :param dataset1: First DataFrame.
        :param dataset2: Second DataFrame.
        :return: Distance between the datasets.
        """

        cols = dataset1.columns
        distance = 0
        for col in cols:
            D, p_value = stats.ks_2samp(dataset1[col], dataset2[col])
            distance = distance + (1 - p_value)
        return distance


class PersonDistanceMetricsOrdering:
    """
    Class to compare two time ordered datasets.
    """

    extreme_value = sys.float_info.max
    tiny_value = 0.000001

    @staticmethod
    def euclidean_distance(dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> float:
        """
        Pair up the datasets and compute the euclidean distances between the sequences of values. Both datasets must
        have the same columns.

        :param dataset1: First DataFrame.
        :param dataset2: Second DataFrame.
        :return: Euclidean distance between all rows in the datasets.
        """

        dist = DistanceMetric.get_metric('euclidean')
        if not len(dataset1.index) == len(dataset2.index):
            return -1
        distance = 0

        for i in range(0, len(dataset1.index)):
            data_row1 = dataset1.iloc[:, i:i + 1].transpose()
            data_row2 = dataset2.iloc[:, i:i + 1].transpose()
            ecl_dist = dist.pairwise(data_row1, data_row2)
            distance = distance + ecl_dist

        return distance

    def lag_correlation_given_lag(self, dataset1: pd.DataFrame, dataset2: pd.DataFrame, lag):
        """
        Compute the distance between two datasets given a set lag. Both datasets must have the same columns.

        :param dataset1: First DataFrame.
        :param dataset2: Second DataFrame.
        :param lag: Lag to use for computing the distance.
        :return:
        """

        distance = 0
        for i in range(0, len(dataset1.columns)):
            # Consider the lengths of the series, and compare the number of points in the smallest series.
            length_ds1 = len(dataset1.index)
            length_ds2 = len(dataset2.index) - lag
            length_used = min(length_ds1, length_ds2)
            if length_used < 1:
                return self.extreme_value
            # Multiply the values as expressed in the book
            ccc = np.multiply(dataset1.ix[0:length_used, i].values, dataset2.ix[lag:length_used + lag, i].values)
            # Add the sum of the mutliplications to the distance and correct for the difference in length
            distance = distance + (float(1) / (float(max(ccc.sum(), self.tiny_value)))) / length_used
        return distance

    def lag_correlation(self, dataset1: pd.DataFrame, dataset2: pd.DataFrame, max_lag: int):
        """
        Compute the lag correlation between two datasets given a set maxsimum lag. For this the best lag is found.

        :param dataset1: First DataFrame.
        :param dataset2: Second DataFrame.
        :param max_lag: Maximum lag so use when finding the best distance.
        :return: Minimum distance between the two given datasets.
        """

        best_dist = -1
        best_lag = 0
        for i in range(0, max_lag + 1):
            # Compute the distance given a lag
            current_dist = self.lag_correlation_given_lag(dataset1, dataset2, i)
            if current_dist < best_dist or best_dist == -1:
                best_dist = current_dist
                best_lag = i
        return best_dist

    def dynamic_time_warping(self, dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> float:
        """
        Simple implementation of the dynamic time warping using the euclidean distance. The implementation follows the
        algorithm explained in the book very closely.

        :param dataset1: First DataFrame.
        :param dataset2: Second DataFrame.
        :return: Cheapest path for time warping.
        """

        # Create a distance matrix between all time points.
        cheapest_path = np.full((len(dataset1.index), len(dataset2.index)), self.extreme_value)
        cheapest_path[0, 0] = 0

        for i in range(1, len(dataset1.index)):
            for j in range(1, len(dataset2.index)):
                data_row1 = dataset1.iloc[i:i + 1, :]
                data_row2 = dataset2.iloc[j:j + 1, :]
                d = sklearn.metrics.pairwise.euclidean_distances(data_row1, data_row2)
                cheapest_path[i, j] = d + min(cheapest_path[i - 1, j], cheapest_path[i, j - 1],
                                              cheapest_path[i - 1, j - 1])
        return cheapest_path[len(dataset1.index) - 1, len(dataset2.index) - 1]
