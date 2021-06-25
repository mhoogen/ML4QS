##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3: Handling Noise and Missing Values            #
#                                                            #
##############################################################

import sys
import copy
from pathlib import Path
import pandas as pd
import numpy as np
from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters
from tqdm import tqdm
import argparse

# Set up the file names and locations
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = sys.argv[1] if len(sys.argv) > 1 else 'chapter3_result_outliers.csv'
RESULT_FNAME = sys.argv[2] if len(sys.argv) > 2 else 'chapter3_result_final.csv'
ORIG_DATASET_FNAME = sys.argv[3] if len(sys.argv) > 3 else 'chapter2_result.csv'


def print_flags():
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    # Import the data from the specified location and parse the date index
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # Create an instance of our visualization class to plot the results
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000

    # Create objects for value imputation, low pass filter and PCA
    MisVal = ImputationMissingValues()
    LowPass = LowPassFilter()
    PCA = PrincipalComponentAnalysis()

    if FLAGS.mode == 'imputation':
        # Impute the missing values and plot an example
        imputed_mean_dataset = MisVal.impute_mean(dataset=copy.deepcopy(dataset), col='hr_watch_rate')
        imputed_median_dataset = MisVal.impute_median(dataset=copy.deepcopy(dataset), col='hr_watch_rate')
        imputed_interpolation_dataset = MisVal.impute_interpolate(dataset=copy.deepcopy(dataset), col='hr_watch_rate')
        DataViz.plot_imputed_values(dataset, ['original', 'mean', 'median', 'interpolation'], 'hr_watch_rate',
                                    imputed_mean_dataset['hr_watch_rate'], imputed_median_dataset['hr_watch_rate'],
                                    imputed_interpolation_dataset['hr_watch_rate'])

    elif FLAGS.mode == 'kalman':
        # Using the result from Chapter 2, try the Kalman filter on the light_phone_lux attribute and study the result
        try:
            original_dataset = pd.read_csv(DATA_PATH / ORIG_DATASET_FNAME, index_col=0)
            original_dataset.index = pd.to_datetime(original_dataset.index)
        except IOError as e:
            print('File not found, try to run previous crowdsignals scripts first!')
            raise e
        KalFilter = KalmanFilters()
        kalman_dataset = KalFilter.apply_kalman_filter(data_table=original_dataset, col='acc_phone_x')
        DataViz.plot_imputed_values(kalman_dataset, ['original', 'kalman'], 'acc_phone_x',
                                    kalman_dataset['acc_phone_x_kalman'])
        DataViz.plot_dataset(data_table=kalman_dataset, columns=['acc_phone_x', 'acc_phone_x_kalman'],
                             match=['exact', 'exact'], display=['line', 'line'])

    elif FLAGS.mode == 'lowpass':
        # Apply a lowpass filter and reduce the importance of the data above 1.5 Hz
        # Determine the sampling frequency
        fs = float(1000) / milliseconds_per_instance

        # Study acc_phone_x
        new_dataset = LowPass.low_pass_filter(data_table=copy.deepcopy(dataset), col='acc_phone_x',
                                              sampling_frequency=fs,
                                              cutoff_frequency=FLAGS.cutoff, order=10)
        DataViz.plot_dataset(new_dataset.iloc[int(0.4 * len(new_dataset.index)):int(0.43 * len(new_dataset.index)), :],
                             ['acc_phone_x', 'acc_phone_x_lowpass'], ['exact', 'exact'], ['line', 'line'])

    elif FLAGS.mode == 'PCA':
        # First impute again, as PCA can not deal with missing values
        for col in [c for c in dataset.columns if 'label' not in c]:
            dataset = MisVal.impute_interpolate(dataset, col)

        # Determine the PC's for all but the target columns (the labels and the heart rate)
        selected_predictor_cols = [c for c in dataset.columns if
                                   (not ('label' in c)) and (not (c == 'hr_watch_rate'))]
        pc_values = PCA.determine_pc_explained_variance(data_table=dataset, cols=selected_predictor_cols)
        cumulated_variance = np.cumsum(pc_values)

        # Plot the explained variance and cumulated variance
        comp_numbers = np.arange(1, len(pc_values) + 1)
        DataViz.plot_xy(x=[comp_numbers, comp_numbers], y=[pc_values, cumulated_variance],
                        xlabel='principal component number', ylabel='explained variance',
                        ylim=[0, 1], line_styles=['b-', 'r-'], names=['Variance', 'Cumulated variance'])

        # Select 7 as the best number of PC's as this explains most of the variance
        n_pcs = 7
        dataset = PCA.apply_pca(data_table=copy.deepcopy(dataset), cols=selected_predictor_cols, number_comp=n_pcs)

        # Visualize the result of the PC's and the overall final dataset
        DataViz.plot_dataset(dataset, ['pca_', 'label'], ['like', 'like'], ['line', 'points'])
        DataViz.plot_dataset(dataset,
                             ['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_', 'pca_',
                              'label'],
                             ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
                             ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])

    elif FLAGS.mode == 'final':
        # Carry out that operation over all columns except for the label
        print('Imputing missing values.')
        for col in tqdm([c for c in dataset.columns if 'label' not in c]):
            dataset = MisVal.impute_interpolate(dataset=dataset, col=col)

        # Include all measurements that have a form of periodicity and filter them
        periodic_measurements = ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'acc_watch_x', 'acc_watch_y',
                                 'acc_watch_z', 'gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z', 'gyr_watch_x',
                                 'gyr_watch_y', 'gyr_watch_z', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z',
                                 'mag_watch_x', 'mag_watch_y', 'mag_watch_z']

        print('Applying low pass filter on peridic measurements.')
        # Determine the sampling frequency.
        fs = float(1000) / milliseconds_per_instance
        for col in tqdm(periodic_measurements):
            dataset = LowPass.low_pass_filter(data_table=dataset, col=col, sampling_frequency=fs,
                                              cutoff_frequency=FLAGS.cutoff, order=10)
            dataset[col] = dataset[col + '_lowpass']
            del dataset[col + '_lowpass']

        # Use the optimal found parameter n_pcs = 7 to apply PCA to the final dataset
        selected_predictor_cols = [c for c in dataset.columns if
                                   (not ('label' in c)) and (not (c == 'hr_watch_rate'))]
        n_pcs = 7
        dataset = PCA.apply_pca(data_table=copy.deepcopy(dataset), cols=selected_predictor_cols, number_comp=n_pcs)

        # Visualize the final overall dataset
        DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux',
                                       'mag_', 'press_phone_', 'pca_', 'label'],
                             ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
                             ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])

        # Store the final outcome
        dataset.to_csv(DATA_PATH / RESULT_FNAME)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help='Select what version to run: final, imputation, lowpass, kalman or PCA \
                            "lowpass" applies the lowpass-filter to a single variable \
                            "imputation" is used for the next chapter \
                            "kalman" applies kalman filter to a single variable \
                            "PCA" is to study the effect of PCA and plot the results \
                            "final" is used for the next chapter',
                        choices=['lowpass', 'imputation', 'kalman', 'PCA', 'final'])
    parser.add_argument('--cutoff', type=float, default=1.5,
                        help='Cutoff frequency for lowpass filter')
    FLAGS, unparsed = parser.parse_known_args()

    # Print args and run main script
    print_flags()
    main()
