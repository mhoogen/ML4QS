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
import copy
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    # Set up file names and locations
    DATA_PATH = Path('./intermediate_datafiles/')
    DATASET_FILENAME = sys.argv[1] if len(sys.argv) > 1 else 'chapter2_result.csv'
    RESULT_FILENAME = sys.argv[2] if len(sys.argv) > 2 else 'chapter3_result_outliers.csv'

    # Import the data from the specified location and parse the date index
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FILENAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run the preceding crowdsignals scripts first!')
        raise e

    # Create an instance of visualization class to plot the results
    DataViz = VisualizeDataset(module_path=__file__)

    # Step 1: See whether there are some outliers that need to be preferably removed

    # Set the columns to experiment on
    outlier_columns = ['acc_phone_x', 'light_phone_lux']

    # Create the outlier classes
    OutlierDistribution = DistributionBasedOutlierDetection()
    OutlierDistance = DistanceBasedOutlierDetection()

    # Investigate the approaches for the relevant attributes
    for col in outlier_columns:
        print(f"Applying outlier criteria for column {col}")

        # Try out all different approaches. The parameters for each approach have been optimized by visual inspection
        dataset = OutlierDistribution.chauvenet(data_table=dataset, col=col)
        DataViz.plot_binary_outliers(data_table=dataset, col=col, outlier_col=f'{col}_outlier')
        dataset = OutlierDistribution.mixture_model(data_table=dataset, col=col, components=3)
        DataViz.plot_dataset(data_table=dataset, columns=[col, col + '_mixture'], match=['exact', 'exact'],
                             display=['line', 'points'])

        # This step requires:
        # n_data_points * n_data_points * point_size = 31839 * 31839 * 32 bits = ~4GB available memory
        try:
            dataset = OutlierDistance.simple_distance_based(data_table=dataset, cols=[col], d_function='euclidean',
                                                            d_min=0.10, f_min=0.99)
            DataViz.plot_binary_outliers(data_table=dataset, col=col, outlier_col='simple_dist_outlier')
        except MemoryError:
            print('Not enough memory available for simple distance-based outlier detection...')
            print('Skipping.')

        try:
            dataset = OutlierDistance.local_outlier_factor(data_table=dataset, cols=[col], d_function='euclidean', k=5)
            DataViz.plot_dataset(data_table=dataset, columns=[col, 'lof'], match=['exact', 'exact'],
                                 display=['line', 'points'])
        except MemoryError:
            print('Not enough memory available for local outlier factor...')
            print('Skipping.')

        # Remove all the stuff from the dataset again
        cols_to_remove = [col + '_outlier', col + '_mixture', 'simple_dist_outlier', 'lof']
        for to_remove in cols_to_remove:
            if to_remove in dataset:
                del dataset[to_remove]

    # Take Chauvenet's criterion and apply it to all but the label data
    for col in [c for c in dataset.columns if 'label' not in c]:
        print(f'Measurement is now: {col}')
        dataset = OutlierDistribution.chauvenet(data_table=dataset, col=col)
        dataset.loc[dataset[f'{col}_outlier'], col] = np.nan
        del dataset[col + '_outlier']

    dataset.to_csv(DATA_PATH / RESULT_FILENAME)


if __name__ == '__main__':
    main()
