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
dataset = pd.read_csv(dataset_path + 'chapter2_result.csv', index_col=0)
dataset.index = dataset.index.to_datetime()

# Computer the number of milliseconds covered by an instane based on the first two rows
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
    print '====', col, '===='
    dataset = OutlierDistr.chauvenet(dataset, col)
    DataViz.plot_binary_outliers(dataset, col, col + '_outlier')
    print 'chauvenet outliers ', dataset[col + '_outlier'].sum()
    dataset = OutlierDistr.mixture_model(dataset, col)
    DataViz.plot_dataset(dataset, [col, col + '_mixture'], ['exact','exact'], ['line', 'points'])
    dataset = OutlierDist.simple_distance_based(dataset, [col], 'euclidean', 0.10, 0.99)
    DataViz.plot_binary_outliers(dataset, col, 'simple_dist_outlier')
    print 'simple dist outliers ', dataset['simple_dist_outlier'].sum()
    dataset = OutlierDist.local_outlier_factor(dataset, [col], 'euclidean', 5)
    DataViz.plot_dataset(dataset, [col, 'lof'], ['exact','exact'], ['line', 'points'])

    # Remove all the stuff from the dataset again.
    del dataset[col + '_outlier']
    del dataset[col + '_mixture']
    del dataset['simple_dist_outlier']
    del dataset['lof']

# We take Chauvent's criterion and apply it to all but the label data...

for col in [c for c in dataset.columns if not 'label' in c]:
    print 'Measurement is now: ' , col
    dataset = OutlierDistr.chauvenet(dataset, col)
    dataset.loc[dataset[col + '_outlier'] == True, col] = np.nan
    del dataset[col + '_outlier']

dataset.to_csv(dataset_path + 'chapter3_result_outliers.csv')