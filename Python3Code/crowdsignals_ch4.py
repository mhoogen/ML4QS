##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4: Feature Engineering                          #
#                                                            #
##############################################################

import sys
import copy
import pandas as pd
from pathlib import Path
from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation

# Set up the file names and locations
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = sys.argv[1] if len(sys.argv) > 1 else 'chapter3_result_final.csv'
RESULT_FNAME = sys.argv[2] if len(sys.argv) > 2 else 'chapter4_result.csv'


def main():
    # Read the result from the previous chapter convert the index to datetime
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # Create an instance of visualization class to plot the results
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds / 1000

    # Chapter 4: Identifying aggregate attributes

    # Focus on the time domain first

    # Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 5 minutes
    window_sizes = [int(float(5000) / milliseconds_per_instance), int(float(0.5 * 60000) / milliseconds_per_instance),
                    int(float(5 * 60000) / milliseconds_per_instance)]

    NumAbs = NumericalAbstraction()
    dataset_copy = copy.deepcopy(dataset)
    for ws in window_sizes:
        print(f'Abstracting numerical features for window size {ws*milliseconds_per_instance/1000}s.')
        dataset_copy = NumAbs.abstract_numerical(data_table=dataset_copy, cols=['acc_phone_x'], window_size=ws,
                                                 aggregation_function='mean')
        dataset_copy = NumAbs.abstract_numerical(data_table=dataset_copy, cols=['acc_phone_x'], window_size=ws,
                                                 aggregation_function='std')

    DataViz.plot_dataset(data_table=dataset_copy,
                         columns=['acc_phone_x', 'acc_phone_x_temp_mean', 'acc_phone_x_temp_std', 'label'],
                         match=['exact', 'like', 'like', 'like'], display=['line', 'line', 'line', 'points'])

    ws = int(float(0.5 * 60000) / milliseconds_per_instance)
    selected_predictor_cols = [c for c in dataset.columns if 'label' not in c]
    print('Calculating mean and std for selected predictor cols.')
    dataset = NumAbs.abstract_numerical(data_table=dataset, cols=selected_predictor_cols, window_size=ws,
                                        aggregation_function='mean')
    dataset = NumAbs.abstract_numerical(data_table=dataset, cols=selected_predictor_cols, window_size=ws,
                                        aggregation_function='std')

    DataViz.plot_dataset(data_table=dataset, columns=['acc_phone_x', 'gyr_phone_x', 'hr_watch_rate', 'light_phone_lux',
                                                      'mag_phone_x', 'press_phone_', 'pca_1', 'label'],
                         match=['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
                         display=['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])

    CatAbs = CategoricalAbstraction()
    print('Abstracting categorical features.')
    dataset = CatAbs.abstract_categorical(data_table=dataset, cols=['label'], match=['like'], min_support=0.03,
                                          window_size=int(float(5 * 60000) / milliseconds_per_instance),
                                          max_pattern_size=2)

    # Move to the frequency domain with the same window size
    FreqAbs = FourierTransformation()
    fs = 1000.0 / milliseconds_per_instance

    periodic_predictor_cols = ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'acc_watch_x', 'acc_watch_y', 'acc_watch_z',
                               'gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z', 'gyr_watch_x', 'gyr_watch_y', 'gyr_watch_z',
                               'mag_phone_x', 'mag_phone_y', 'mag_phone_z', 'mag_watch_x', 'mag_watch_y', 'mag_watch_z']
    data_table = FreqAbs.abstract_frequency(data_table=copy.deepcopy(dataset), cols=['acc_phone_x'],
                                            window_size=int(float(10000) / milliseconds_per_instance), sampling_rate=fs)

    # Spectral analysis
    DataViz.plot_dataset(data_table=data_table,
                         columns=['acc_phone_x_max_freq', 'acc_phone_x_freq_weighted', 'acc_phone_x_pse', 'label'],
                         match=['like', 'like', 'like', 'like'], display=['line', 'line', 'line', 'points'])

    print('Abstracting frequency features.')
    dataset = FreqAbs.abstract_frequency(data_table=dataset, cols=periodic_predictor_cols,
                                         window_size=int(float(10000) / milliseconds_per_instance), sampling_rate=fs)

    # Take a certain percentage of overlap in the windows, otherwise training examples will be too much alike.
    # Set the allowed percentage of overlap
    window_overlap = 0.9
    skip_points = int((1 - window_overlap) * ws)
    dataset = dataset.iloc[::skip_points, :]

    DataViz.plot_dataset(data_table=dataset, columns=['acc_phone_x', 'gyr_phone_x', 'hr_watch_rate', 'light_phone_lux',
                                                      'mag_phone_x', 'press_phone_', 'pca_1', 'label'],
                         match=['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
                         display=['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])

    # Store the generated dataset
    dataset.to_csv(DATA_PATH / RESULT_FNAME)


if __name__ == '__main__':
    main()
