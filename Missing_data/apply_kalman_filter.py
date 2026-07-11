import pandas as pd
import numpy as np
from pykalman import KalmanFilter

New_weather_steps = pd.read_csv("Transformed_weather_steps.csv")
New_weather_steps = New_weather_steps.drop("Unnamed: 0",axis=1)

# Implements the Kalman filter for single columns.
def apply_kalman_filter(data_table, col):
    # Initialize the Kalman filter with the trivial transition and observation matrices.
    kf = KalmanFilter(transition_matrices=[[1]], observation_matrices=[[1]])

    numpy_array_state = data_table[col].values
    numpy_array_state = numpy_array_state.astype(np.float32)
    numpy_matrix_state_with_mask = np.ma.masked_invalid(numpy_array_state)

    # Find the best other parameters based on the data (e.g. Q)
    kf = kf.em(numpy_matrix_state_with_mask, n_iter=5)

    # And apply the filter.
    (new_data, filtered_state_covariances) = kf.filter(numpy_matrix_state_with_mask)

    data_table[col + '_kalman'] = new_data
    return(data_table)

data_table = New_weather_steps
col = "steps"
data_table = apply_kalman_filter(data_table, col)