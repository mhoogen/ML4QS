##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 5                                               #
#                                                            #
##############################################################

import math
import numbers
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import stats
import sys
from sklearn.neighbors import DistanceMetric
import sklearn



# Class defining the distance metrics that are not available as standard ones....
class InstanceDistanceMetrics:

    # S for gowers distance
    def s(self, val1, val2, range):
        # If we compare numbers we look at the difference and normalize.
        if isinstance(val1, numbers.Number) and isinstance(val1, numbers.Number):
            return 1 - (float(abs(val1-val2))/range)
        # If we compare something else, we just look at whether they are equal.
        else:
            if val1 == val2:
                return 1
            else:
                return 0

    # Delta for gowers distance.
    def delta(self, val1, val2):
        # Check whether both values are known (i.e. nan), if so the delta is 1, 0 otherwise.
        if (not np.isnan(val1)) and (not np.isnan(val2)):
            return 1
        return 0

    # Define gowers distance between two rows, given the ranges of the variables
    # over the entire dataset (over all columns in row1 and row2)
    def gowers_similarity(self, data_row1, data_row2, ranges):
        # We cannot computer if the lengths are not equal.
        if len(data_row1.columns) != len(data_row2.columns):
            return -1

        delta_total = 0
        s_total = 0

        # iterate over all columns.
        for i in range(0, len(data_row1.columns)):
            val1 = data_row1[data_row1.columns[i]].values[0]
            val2 = data_row2[data_row2.columns[i]].values[0]
            # compute the delta
            delta = self.delta(val1, val2)
            delta_total = delta_total + delta
            if delta > 0:
                # and compute the s if the delta is above 0.
                s_total = s_total + self.s(val1, val2, ranges[i])
        return float(s_total)/delta_total

# Class to flatten datasets or compute the statistical difference between cases.
class PersonDistanceMetricsNoOrdering:

    gower = 'gower'
    minkowski = 'minkowski'

    # This returns a dataset with aggregated data instances based on the mean values
    # in the rows.
    def create_instances_mean(self, datasets):
        index = range(0, len(datasets))
        cols = datasets[0].columns
        new_dataset = pd.DataFrame(index=index, columns=cols)

        for i in range(0, len(datasets)):
            for col in cols:
                # Compute the mean per column and assign that
                # value for the row representing the current
                # dataset.
                new_dataset.ix[i, col] = datasets[i][col].mean()

        return new_dataset

    # Fit datasets to normal distribution and use parameters as instances
    def create_instances_normal_distribution(self, datasets):
        index = range(0, len(datasets))
        cols = datasets[0].columns
        new_cols = []
        # Create new columns for the parameters of the distribution.
        for col in cols:
            new_cols.append(col + '_mu')
            new_cols.append(col + '_sigma')
        new_dataset = pd.DataFrame(index=index, columns=new_cols)

        for i in range(0, len(datasets)):
            for col in cols:
                # Fit the distribution and assign the values to the
                # row representing the dataset.
                mu, sigma = norm.fit(datasets[i][col])
                new_dataset.ix[i, col + '_mu'] = mu
                new_dataset.ix[i, col + '_sigma'] = sigma

        return new_dataset

    # This defines the distance between datasets based on the statistical
    # differences between the distribution we can only compute
    # distances pairwise.
    def p_distance(self, dataset1, dataset2):

        cols = dataset1.columns
        distance = 0
        for col in cols:
            D, p_value = stats.ks_2samp(dataset1[col], dataset2[col])
            distance= distance + (1-p_value)
        return distance

# Class to compare two time ordered datasets.
class PersonDistanceMetricsOrdering:

    extreme_value = sys.float_info.max
    tiny_value = 0.000001

    # Directly pair up the datasets and computer the euclidean
    # distances between the sequences of values.
    def euclidean_distance(self, dataset1, dataset2):
        dist = DistanceMetric.get_metric('euclidean')
        if not len(dataset1.index) == len(dataset2.index):
            return -1
        distance = 0

        for i in range(0, len(dataset1.index)):
            data_row1 = dataset1.iloc[:,i:i+1].transpose()
            data_row2 = dataset2.iloc[:,i:i+1].transpose()
            ecl_dist = dist.pairwise(data_row1, data_row2)
            distance = distance + ecl_dist

        return distance

    # Compute the distance between two datasets given a set lag.
    def lag_correlation_given_lag(self, dataset1, dataset2, lag):
        distance = 0
        for i in range(0, len(dataset1.columns)):
            # consider the lengths of the series, and compare the
            # number of points in the smallest series.
            length_ds1 = len(dataset1.index)
            length_ds2 = len(dataset2.index) - lag
            length_used = min(length_ds1, length_ds2)
            if length_used < 1:
                return self.extreme_value
            # We multiply the values as expressed in the book.
            ccc = np.multiply(dataset1.ix[0:length_used, i].values, dataset2.ix[lag:length_used+lag, i].values)
            # We add the sum of the mutliplications to the distance. Correct for the difference in length.
            distance = distance + (float(1)/(float(max(ccc.sum(), self.tiny_value))))/length_used
        return distance

    # Compute the lag correlation. For this we find the best lag.
    def lag_correlation(self, dataset1, dataset2, max_lag):
        best_dist = -1
        best_lag = 0
        for i in range(0, max_lag+1):
            # Compute the distance given a lag.
            current_dist = self.lag_correlation_given_lag(dataset1, dataset2, i)
            if current_dist < best_dist or best_dist == -1:
                best_dist = current_dist
                best_lag = i
        return best_dist

    # Simple implementation of the dtw. Note that we use the euclidean distance here..
    # The implementation follows the algorithm explained in the book very closely.
    def dynamic_time_warping(self, dataset1, dataset2):
        # Create a distance matrix between all time points.
        cheapest_path = np.full((len(dataset1.index), len(dataset2.index)), self.extreme_value)
        cheapest_path[0,0] = 0
        DM = InstanceDistanceMetrics()


        for i in range(1, len(dataset1.index)):
            for j in range(1, len(dataset2.index)):
                data_row1 = dataset1.iloc[i:i+1,:]
                data_row2 = dataset2.iloc[j:j+1,:]
                d = sklearn.metrics.pairwise.euclidean_distances(data_row1, data_row2)
                cheapest_path[i,j] = d + min(cheapest_path[i-1, j], cheapest_path[i, j-1], cheapest_path[i-1, j-1])
        return cheapest_path[len(dataset1.index)-1, len(dataset2.index)-1]

