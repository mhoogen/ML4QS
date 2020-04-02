##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import scipy
import math
from sklearn import mixture
import numpy as np
import pandas as pd
import util.util as util
import copy

# Class for outlier detection algorithms based on some distribution of the data. They
# all consider only single points per row (i.e. one column).
class DistributionBasedOutlierDetection:

    # Finds outliers in the specified column of datatable and adds a binary column with
    # the same name extended with '_outlier' that expresses the result per data point.
    def chauvenet(self, data_table, col):
        # Taken partly from: https://www.astro.rug.nl/software/kapteyn/

        # Computer the mean and standard deviation.
        mean = data_table[col].mean()
        std = data_table[col].std()
        N = len(data_table.index)
        criterion = 1.0/(2*N)

        # Consider the deviation for the data points.
        deviation = abs(data_table[col] - mean)/std

        # Express the upper and lower bounds.
        low = -deviation/math.sqrt(2)
        high = deviation/math.sqrt(2)
        prob = []
        mask = []

        # Pass all rows in the dataset.
        for i in range(0, len(data_table.index)):
            # Determine the probability of observing the point
            prob.append(1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i])))
            # And mark as an outlier when the probability is below our criterion.
            mask.append(prob[i] < criterion)
        data_table[col + '_outlier'] = mask
        return data_table

    # Fits a mixture model towards the data expressed in col and adds a column with the probability
    # of observing the value given the mixture model.
    def mixture_model(self, data_table, col):
        # Fit a mixture model to our data.
        data = data_table[data_table[col].notnull()][col]
        g = mixture.GMM(n_components=3, n_iter=1)

        g.fit(data.reshape(-1,1))

        # Predict the probabilities
        probs = g.score(data.reshape(-1,1))

        # Create the right data frame and concatenate the two.
        data_probs = pd.DataFrame(np.power(10, probs), index=data.index, columns=[col+'_mixture'])
        data_table = pd.concat([data_table, data_probs], axis=1)

        return data_table

# Class for distance based outlier detection.
class DistanceBasedOutlierDetection:


    # Create distance table between rows in the data table. Here, only cols are considered and the specified
    # distance function is used to compute the distance.
    def distance_table(self, data_table, cols, d_function):
        return pd.DataFrame(scipy.spatial.distance.squareform(util.distance(data_table.ix[:, cols], d_function)), columns=data_table.index, index=data_table.index)

    # The most simple distance based algorithm. We assume a distance function, e.g. 'euclidean'
    # and a minimum distance of neighboring points and frequency of occurrence.
    def simple_distance_based(self, data_table, cols, d_function, dmin, fmin):
        # Normalize the dataset first.
        new_data_table = util.normalize_dataset(data_table.dropna(axis=0, subset=cols), cols)
        # Create the distance table first between all instances:
        distances = self.distance_table(new_data_table, cols, d_function)

        mask = []
        # Pass the rows in our table.
        for i in range(0, len(new_data_table.index)):
            # Check what faction of neighbors are beyond dmin.
            frac = (float(sum([1 for col_val in distances.ix[i,:].tolist() if col_val > dmin]))/len(new_data_table.index))
            # Mark as an outlier if beyond the minimum frequency.
            mask.append(frac > fmin)
        data_mask = pd.DataFrame(mask, index=new_data_table.index, columns=['simple_dist_outlier'])
        data_table = pd.concat([data_table, data_mask], axis=1)
        return data_table

    # Computes the local outlier factor. K is the number of neighboring points considered, d_function
    # the distance function again (e.g. 'euclidean').
    def local_outlier_factor(self, data_table, cols, d_function, k):
        # Inspired on https://github.com/damjankuznar/pylof/blob/master/lof.py
        # but tailored towards the distance metrics and data structures used here.

        # Normalize the dataset first.
        new_data_table = util.normalize_dataset(data_table.dropna(axis=0, subset=cols), cols)
        # Create the distance table first between all instances:
        self.distances = self.distance_table(new_data_table, cols, d_function)

        outlier_factor = []
        # Compute the outlier score per row.
        for i in range(0, len(new_data_table.index)):
            print i
            outlier_factor.append(self.local_outlier_factor_instance(i, k))
        data_outlier_probs = pd.DataFrame(outlier_factor, index=new_data_table.index, columns=['lof'])
        data_table = pd.concat([data_table, data_outlier_probs], axis=1)
        return data_table

    # The distance between a row i1 and i2.
    def reachability_distance(self, k, i1, i2):
        # Compute the k-distance of i2.
        k_distance_value, neighbors = self.k_distance(i2, k)
        # The value is the max of the k-distance of i2 and the real distance.
        return max([k_distance_value, self.distances.ix[i1,i2]])

    # Compute the local reachability density for a row i, given a k-distance and set of neighbors.
    def local_reachability_density(self, i, k, k_distance_i, neighbors_i):
        # Set distances to neighbors to 0.
        reachability_distances_array = [0]*len(neighbors_i)

        # Compute the reachability distance between i and all neighbors.
        for i, neighbor in enumerate(neighbors_i):
            reachability_distances_array[i] = self.reachability_distance(k, i, neighbor)
        if not any(reachability_distances_array):
            return float("inf")
        else:
            # Return the number of neighbors divided by the sum of the reachability distances.
            return len(neighbors_i) / sum(reachability_distances_array)

    # Compute the k-distance of a row i, namely the maximum distance within the k nearest neighbors
    # and return a tuple containing this value and the neighbors within this distance.
    def k_distance(self, i, k):
        # Simply look up the values in the distance table, select the min_pts^th lowest value and take the value pairs
        # Take min_pts + 1 as we also have the instance itself in there.
        neighbors = np.argpartition(np.array(self.distances.ix[i,:]), k+1)[0:(k+1)].tolist()
        if i in neighbors:
            neighbors.remove(i)
        return max(self.distances.ix[i,neighbors]), neighbors

    # Compute the local outlier score of our row i given a setting for k.
    def local_outlier_factor_instance(self, i, k):
        # Compute the k-distance for i.
        k_distance_value, neighbors = self.k_distance(i, k)
        # Computer the local reachability given the found k-distance and neighbors.
        instance_lrd = self.local_reachability_density(i, k, k_distance_value, neighbors)
        lrd_ratios_array = [0]* len(neighbors)

        # Computer the k-distances and local reachability density of the neighbors
        for i, neighbor in enumerate(neighbors):
            k_distance_value_neighbor, neighbors_neighbor = self.k_distance(neighbor, k)
            neighbor_lrd = self.local_reachability_density(neighbor, k, k_distance_value_neighbor, neighbors_neighbor)
            # Store the ratio between the neighbor and the row i.
            lrd_ratios_array[i] = neighbor_lrd / instance_lrd

        # Return the average ratio.
        return sum(lrd_ratios_array) / len(neighbors)

