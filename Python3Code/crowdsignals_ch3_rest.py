##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import sys
import copy
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters

# Set up the file names and locations.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = sys.argv[1] if len(sys.argv) > 1 else 'chapter3_result_outliers.csv'
RESULT_FNAME = sys.argv[2] if len(sys.argv) > 2 else 'chapter3_result_final.csv'
ORIG_DATASET_FNAME = sys.argv[3] if len(sys.argv) > 3 else 'chapter2_result.csv'

# Next, import the data from the specified location and parse the date index.
try:
    dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

# We'll create an instance of our visualization class to plot the results.
DataViz = VisualizeDataset(__file__)

# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000 # todo: I believe this command doesn't work for a whole-second interval (returns 0ms)

# Let us impute the missing values and plot an example.

MisVal = ImputationMissingValues()
imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'hr_watch_rate')
imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'hr_watch_rate')
imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'hr_watch_rate')
DataViz.plot_imputed_values(dataset, ['original', 'mean', 'interpolation'], 'hr_watch_rate', imputed_mean_dataset['hr_watch_rate'], imputed_interpolation_dataset['hr_watch_rate'])

# Now, let us carry out that operation over all columns except for the label.

for col in [c for c in dataset.columns if not 'label' in c]:
    dataset = MisVal.impute_interpolate(dataset, col)

# Using the result from Chapter 2, let us try the Kalman filter on the light_phone_lux attribute and study the result.

original_dataset = pd.read_csv(DATA_PATH / ORIG_DATASET_FNAME, index_col=0)
original_dataset.index = pd.to_datetime(original_dataset.index)
KalFilter = KalmanFilters()
kalman_dataset = KalFilter.apply_kalman_filter(original_dataset, 'acc_phone_x')
DataViz.plot_imputed_values(kalman_dataset, ['original', 'kalman'], 'acc_phone_x', kalman_dataset['acc_phone_x_kalman'])
DataViz.plot_dataset(kalman_dataset, ['acc_phone_x', 'acc_phone_x_kalman'], ['exact','exact'], ['line', 'line'])

# We ignore the Kalman filter output for now...

# Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

LowPass = LowPassFilter()

# Determine the sampling frequency.
fs = float(1000)/milliseconds_per_instance
cutoff = 1.5

# Let us study acc_phone_x:
new_dataset = LowPass.low_pass_filter(copy.deepcopy(dataset), 'acc_phone_x', fs, cutoff, order=10)
DataViz.plot_dataset(new_dataset.iloc[int(0.4*len(new_dataset.index)):int(0.43*len(new_dataset.index)), :],
                     ['acc_phone_x', 'acc_phone_x_lowpass'], ['exact','exact'], ['line', 'line'])

# And not let us include all measurements that have a form of periodicity (and filter them):
periodic_measurements = ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'acc_watch_x', 'acc_watch_y', 'acc_watch_z', 'gyr_phone_x', 'gyr_phone_y',
                         'gyr_phone_z', 'gyr_watch_x', 'gyr_watch_y', 'gyr_watch_z', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z', 'mag_watch_x',
                         'mag_watch_y', 'mag_watch_z']

for col in periodic_measurements:
    dataset = LowPass.low_pass_filter(dataset, col, fs, cutoff, order=10)
    dataset[col] = dataset[col + '_lowpass']
    del dataset[col + '_lowpass']


# Determine the PC's for all but our target columns (the labels and the heart rate)
# We simplify by ignoring both, we could also ignore one first, and apply a PC to the remainder.

PCA = PrincipalComponentAnalysis()
selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c)) and (not (c == 'hr_watch_rate'))]
pc_values = PCA.determine_pc_explained_variance(dataset, selected_predictor_cols)

# Plot the variance explained.
DataViz.plot_xy(x=[range(1, len(selected_predictor_cols)+1)], y=[pc_values],
                xlabel='principal component number', ylabel='explained variance',
                ylim=[0,1], line_styles=['b-'])

# We select 7 as the best number of PC's as this explains most of the variance

n_pcs = 7

dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)

#And we visualize the result of the PC's

DataViz.plot_dataset(dataset, ['pca_', 'label'], ['like', 'like'], ['line', 'points'])

# And the overall final dataset:

DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_', 'pca_', 'label'],
                     ['like', 'like', 'like', 'like', 'like', 'like', 'like','like', 'like'],
                     ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])

# Store the outcome.

dataset.to_csv(DATA_PATH / RESULT_FNAME)