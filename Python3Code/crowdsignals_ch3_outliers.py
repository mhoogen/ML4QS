##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3: Handling Noise and Missing Values            #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Set up file names and locations
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FILENAME = sys.argv[1] if len(sys.argv) > 1 else 'chapter2_result.csv'
RESULT_FILENAME = sys.argv[2] if len(sys.argv) > 2 else 'chapter3_result_outliers.csv'


def print_flags():
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    # Import the data from the specified location and parse the date index
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FILENAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run the preceding crowdsignals scripts first!')
        raise e

    # Create an instance of visualization class to plot the results
    DataViz = VisualizeDataset(module_path=__file__)
    # Create the outlier classes
    OutlierDistribution = DistributionBasedOutlierDetection()
    OutlierDistance = DistanceBasedOutlierDetection()

    # Step 1: If requested, see whether there are some outliers that need to be preferably removed
    # Set the columns to experiment on
    outlier_columns = ['acc_phone_x', 'light_phone_lux']

    if FLAGS.mode == 'chauvenet':
        # Investigate the outlier columns using chauvenet criterium
        for col in outlier_columns:

            print(f"Applying Chauvenet outlier criteria for column {col}")

            # And try out all different approaches. Note that we have done some optimization
            # of the parameter values for each of the approaches by visual inspection.
            dataset = OutlierDistr.chauvenet(dataset, col, FLAGS.C)
            DataViz.plot_binary_outliers(
                dataset, col, col + '_outlier')

    elif FLAGS.mode == 'mixture':
        # Investigate the outlier columns using mixture models
        for col in outlier_columns:
            print(f"Applying mixture model for column {col}")
            dataset = OutlierDistribution.mixture_model(data_table=dataset, col=col, components=3)
            DataViz.plot_dataset(data_table=dataset, columns=[col, f'{col}_mixture'], match=['exact', 'exact'],
                                 display=['line', 'points'])

    elif FLAGS.mode == 'distance':
        for col in outlier_columns:
            print(f"Applying distance based outlier detection for column {col}")
            # This step requires:
            # n_data_points * n_data_points * point_size = 31839 * 31839 * 32 bits = ~4GB available memory
            try:
                dataset = OutlierDistance.simple_distance_based(data_table=dataset, cols=[col], d_function='euclidean',
                                                                d_min=FLAGS.dmin, f_min=FLAGS.fmin)
                DataViz.plot_binary_outliers(data_table=dataset, col=col, outlier_col='simple_dist_outlier')
            except MemoryError:
                print('Not enough memory available for simple distance-based outlier detection...')
                print('Skipping.')

    elif FLAGS.mode == 'LOF':
        for col in outlier_columns:
            print(f"Applying Local outlier factor for column {col}")
            try:
                dataset = OutlierDistance.local_outlier_factor(data_table=dataset, cols=[col], d_function='euclidean',
                                                               k=FLAGS.K)
                DataViz.plot_dataset(data_table=dataset, columns=[col, 'lof'], match=['exact', 'exact'],
                                     display=['line', 'points'])
            except MemoryError:
                print('Not enough memory available for local outlier factor...')
                print('Skipping.')

    elif FLAGS.mode == 'final':
        # Take Chauvenet's criterion and apply it to all but the label column in the main dataset
        for col in [c for c in dataset.columns if 'label' not in c]:
            print(f'Measurement is now: {col}')

            dataset = OutlierDistribution.chauvenet(data_table=dataset, col=col)
            dataset.loc[dataset[f'{col}_outlier'], col] = np.nan

            del dataset[col + '_outlier']

        dataset.to_csv(DATA_PATH / RESULT_FILENAME)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='final',

                        help="Select what version to run: LOF, distance, mixture, chauvenet or final \
                        'LOF' applies the Local Outlier Factor to a single variable \
                        'distance' applies a distance based outlier detection method to a single variable \
                        'mixture' applies a mixture model to detect outliers for a single variable\
                        'chauvenet' applies Chauvenet outlier detection method to a single variable \
                        'final' is used for the next chapter", choices=['LOF', 'distance', 'mixture', 'chauvenet', 'final'])

    parser.add_argument('--C', type=float, default=2,
                        help="Chauvenet: C parameter")
   
    parser.add_argument('--K', type=int, default=5,
                        help='Local Outlier Factor:  K is the number of neighboring points considered')

    parser.add_argument('--dmin', type=float, default=0.10,
                        help='Simple distance based:  dmin is ... ')

    parser.add_argument('--fmin', type=float, default=0.99,
                        help='Simple distance based:  fmin is ... ')


    FLAGS, unparsed = parser.parse_known_args()

    # Print args and run main script
    print_flags()
    main()
