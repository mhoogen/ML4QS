##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import numpy as np
from pykalman import KalmanFilter

# Implements the Kalman filter for single columns.
class KalmanFilters:

    # Very simple Kalman filter: fill missing values and remove outliers for single attribute.
    # We assume a very simple transition matrix, namely simply a [[1]]. It
    # is however still useful as it is able to dampen outliers and impute missing values. The new
    # values are appended in a new column.
    def apply_kalman_filter(self, data_table, col):

        # Initialize the Kalman filter with the trivial transition and observation matrices.
        kf = KalmanFilter(transition_matrices = [[1]], observation_matrices = [[1]])

        numpy_array_state = data_table.as_matrix(columns=[col])
        numpy_array_state = numpy_array_state.astype(np.float32)
        numpy_matrix_state_with_mask = np.ma.masked_invalid(numpy_array_state)

        # Find the best other parameters based on the data (e.g. Q)
        kf = kf.em(numpy_matrix_state_with_mask, n_iter=5)

        # And apply the filter.
        (new_data, filtered_state_covariances) = kf.filter(numpy_matrix_state_with_mask)

        data_table[col + '_kalman'] = new_data
        return data_table
