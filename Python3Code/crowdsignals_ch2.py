##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys

# Chapter 2: Initial exploration of the dataset.

"""
First, we set some module-level constants to store our data locations. These are saved as a pathlib.Path object, the
preferred way to handle OS paths in Python 3 (https://docs.python.org/3/library/pathlib.html). Using the Path's methods,
you can execute most path-related operations such as making directories.

sys.argv contains a list of keywords entered in the command line, and can be used to specify a file path when running
a script from the command line. For example:

$ python3 crowdsignals_ch2.py my/proj/data/folder my_dataset.csv

If no location is specified, the default locations in the else statement are chosen, which are set to load each script's
output into the next by default.
"""

DATASET_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else './datasets/crowdsignals/csv-participant-one/')
RESULT_PATH = Path('./intermediate_datafiles/')
RESULT_FNAME = sys.argv[2] if len(sys.argv) > 2 else 'chapter2_result.csv'

# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
GRANULARITIES = [60000, 250]

# We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

datasets = []
for milliseconds_per_instance in GRANULARITIES:
    print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

    # Create an initial dataset object with the base directory for our data and a granularity
    data_engineer = CreateDataset(base_dir=DATASET_PATH, granularity=milliseconds_per_instance)

    # Add the selected measurements to it.

    # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    data_engineer.add_numerical_dataset(file='accelerometer_phone.csv', timestamp_col='timestamps',
                                        value_cols=['x', 'y', 'z'], aggregation='avg', prefix='acc_phone_')
    data_engineer.add_numerical_dataset(file='accelerometer_smartwatch.csv', timestamp_col='timestamps',
                                        value_cols=['x', 'y', 'z'], aggregation='avg', prefix='acc_watch_')

    # We add the gyroscope data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    data_engineer.add_numerical_dataset(file='gyroscope_phone.csv', timestamp_col='timestamps',
                                        value_cols=['x', 'y', 'z'], aggregation='avg', prefix='gyr_phone_')
    data_engineer.add_numerical_dataset(file='gyroscope_smartwatch.csv', timestamp_col='timestamps',
                                        value_cols=['x', 'y', 'z'], aggregation='avg', prefix='gyr_watch_')

    # We add the heart rate (continuous numerical measurements) and aggregate by averaging again
    data_engineer.add_numerical_dataset(file='heart_rate_smartwatch.csv', timestamp_col='timestamps',
                                        value_cols=['rate'], aggregation='avg', prefix='hr_watch_')

    # We add the labels provided by the users. These are categorical events that might overlap. We add them
    # as binary attributes (i.e. add a one to the attribute representing the specific value for the label if it
    # occurs within an interval).
    data_engineer.add_event_dataset(file='labels.csv', start_timestamp_col='label_start', end_timestamp_col='label_end',
                                    value_col='label', aggregation='binary')

    # We add the amount of light sensed by the phone (continuous numerical measurements) and aggregate by averaging
    data_engineer.add_numerical_dataset(file='light_phone.csv', timestamp_col='timestamps', value_cols=['lux'],
                                        aggregation='avg', prefix='light_phone_')

    # We add the magnetometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    data_engineer.add_numerical_dataset(file='magnetometer_phone.csv', timestamp_col='timestamps',
                                        value_cols=['x', 'y', 'z'], aggregation='avg', prefix='mag_phone_')
    data_engineer.add_numerical_dataset(file='magnetometer_smartwatch.csv', timestamp_col='timestamps',
                                        value_cols=['x', 'y', 'z'], aggregation='avg', prefix='mag_watch_')

    # We add the pressure sensed by the phone (continuous numerical measurements) and aggregate by averaging again
    data_engineer.add_numerical_dataset(file='pressure_phone.csv', timestamp_col='timestamps', value_cols=['pressure'],
                                        aggregation='avg', prefix='press_phone_')

    # Get the resulting pandas data table
    dataset = data_engineer.data_table

    # Plot the data
    DataViz = VisualizeDataset(__file__)

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset=dataset, cols=['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'acc_watch_x',
                                                        'acc_watch_y', 'acc_watch_z'])

    # Plot all data
    DataViz.plot_dataset(data_table=dataset,
                         columns=['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_', 'label'],
                         match=['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
                         display=['line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])

    # And print a summary of the dataset.
    util.print_statistics(dataset=dataset)
    datasets.append(copy.deepcopy(dataset))

    # If needed, we could save the various versions of the dataset we create in the loop with logical filenames:
    # dataset.to_csv(RESULT_PATH / f'chapter2_result_{milliseconds_per_instance}')

# Make a table like the one shown in the book, comparing the two datasets produced.
util.print_latex_table_statistics_two_datasets(dataset1=datasets[0], dataset2=datasets[1])

# Finally, store the last dataset we generated (250 ms).
dataset.to_csv(RESULT_PATH / RESULT_FNAME)
