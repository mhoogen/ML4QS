##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
import copy
import pandas as pd
import numpy as np

# Let is create our visualization class again.
DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sture the index is of the type datetime.
dataset_path = './intermediate_datafiles/'
try:
    dataset = pd.read_csv(dataset_path + 'chapter2_result.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset.index = dataset.index.to_datetime()

# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

# Step 1: Let us see whether we have some outliers we would prefer to remove.

# Determine the columns we want to experiment on.
outlier_columns = ['acc_phone_x', 'light_phone_lux']

# Create the outlier classes.
OutlierDistr = DistributionBasedOutlierDetection()
OutlierDist = DistanceBasedOutlierDetection()

#And investigate the approaches for all relevant attributes.
for col in outlier_columns:
    # And try out all different approaches. Note that we have done some optimization
    # of the parameter values for each of the approaches by visual inspection.
    dataset = OutlierDistr.chauvenet(dataset, col)
    DataViz.plot_binary_outliers(dataset, col, col + '_outlier')
    dataset = OutlierDistr.mixture_model(dataset, col)
    DataViz.plot_dataset(dataset, [col, col + '_mixture'], ['exact','exact'], ['line', 'points'])
    # This requires:
    # n_data_points * n_data_points * point_size =
    # 31839 * 31839 * 64 bits = ~8GB available memory
    try:
        dataset = OutlierDist.simple_distance_based(dataset, [col], 'euclidean', 0.10, 0.99)
        DataViz.plot_binary_outliers(dataset, col, 'simple_dist_outlier')
    except MemoryError as e:
        print('Not enough memory available for simple distance-based outlier detection...')
        print('Skipping.')
    
    try:
        dataset = OutlierDist.local_outlier_factor(dataset, [col], 'euclidean', 5)
        DataViz.plot_dataset(dataset, [col, 'lof'], ['exact','exact'], ['line', 'points'])
    except MemoryError as e:
        print('Not enough memory available for lof...')
        print('Skipping.')

    # Remove all the stuff from the dataset again.
    cols_to_remove = [col + '_outlier', col + '_mixture', 'simple_dist_outlier', 'lof']
    for to_remove in cols_to_remove:
        if to_remove in dataset:
            del dataset[to_remove]

# We take Chauvent's criterion and apply it to all but the label data...

for col in [c for c in dataset.columns if not 'label' in c]:
    print 'Measurement is now: ' , col
    dataset = OutlierDistr.chauvenet(dataset, col)
    dataset.loc[dataset[col + '_outlier'] == True, col] = np.nan
    del dataset[col + '_outlier']

dataset.to_csv(dataset_path + 'chapter3_result_outliers.csv')
