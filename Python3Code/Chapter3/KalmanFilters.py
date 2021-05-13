##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import numpy as np
import pandas as pd
from pykalman import KalmanFilter


class KalmanFilters:

    @staticmethod
    def apply_kalman_filter(data_table: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Fill missing values and remove outliers for single attribute using a simple Kalman-Filter. We assume a very
        simple transition matrix, namely a [[1]]. It is however still useful as it is able to dampen outliers and impute
         missing values. The imputed values are appended in a new column named 'col_kalman'.

        :param data_table: Dataframe with the data to apply the Kalman filter on.
        :param col: Name of the col with the data to process.
        :return: Original dataframe with new column containing the filtered data.
        """

        # Initialize the Kalman filter with the trivial transition and observation matrices
        kf = KalmanFilter(transition_matrices=[[1]], observation_matrices=[[1]])

        numpy_array_state = data_table[col].values
        numpy_array_state = numpy_array_state.astype(np.float32)
        numpy_matrix_state_with_mask = np.ma.masked_invalid(numpy_array_state)

        # Find the best other parameters based on the data (e.g. Q) and apply the filter
        kf = kf.em(numpy_matrix_state_with_mask, n_iter=5)
        new_data, filtered_state_covariances = kf.filter(numpy_matrix_state_with_mask)

        data_table[col + '_kalman'] = new_data
        return data_table
