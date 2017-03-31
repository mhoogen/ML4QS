from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import FourierTransformation
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.TextAbstraction import TextAbstraction
from Chapter5.DistanceMetrics import InstanceDistanceMetrics
from Chapter5.DistanceMetrics import PersonDistanceMetricsNoOrdering
from Chapter5.DistanceMetrics import PersonDistanceMetricsOrdering
from Chapter5.Clustering import NonHierarchicalClustering
from Chapter5.Clustering import HierarchicalClustering
import copy
import pandas as pd

# Of course we repeat some stuff from Chapter 3, namely to load the dataset

DataViz = VisualizeDataset()
milliseconds_per_instance = 50000

# Create an initial dataset object

DataSet = CreateDataset('/Users/markhoogendoorn/Dropbox/Quantified-Self-Book/datasets/crowdsignals.io/Csv-merged/', milliseconds_per_instance)

DataSet.add_numerical_dataset('merged_accelerometer_1466120659000000000_1466125720000000000.csv', 'timestamps', ['x','y','z'], 'avg', 'phone_acc_')
DataSet.add_event_dataset('merged_interval_label_1466120784502000000_1466125692364000000.csv', 'start', 'end', 'label', 'binary')
#DataSet.add_event_dataset('merged_apps_1466120659731000000_1466125749000000000.csv', 'start', 'end', 'app', 'binary')
#DataSet.add_numerical_dataset('merged_msBandSkinTemperature_1466120660000000000_1466125634000000000.csv', 'timestamps', ['temperature'], 'avg', 'watch_skin_')

dataset = DataSet.data_table
KalFilter = KalmanFilters()

# Apply the Kalman Filter to all
dataset = KalFilter.apply_kalman_filter(dataset, 'phone_acc_x')
dataset = KalFilter.apply_kalman_filter(dataset, 'phone_acc_y')
dataset = KalFilter.apply_kalman_filter(dataset, 'phone_acc_z')

# And remove the original data without the Kalman filter
del dataset['phone_acc_x']
del dataset['phone_acc_y']
del dataset['phone_acc_z']

# Try to computer the distance between two data elements

ranges = []
for col in dataset.columns:
    ranges.append(dataset[col].max() - dataset[col].min())

data_row1 = dataset.iloc[0:1,:]
data_row2 = dataset.iloc[1:2,:]
DM = InstanceDistanceMetrics()

print data_row1
print data_row2
distance = DM.gowers_similarity(data_row1, data_row2, ranges)
print distance

DM = PersonDistanceMetricsNoOrdering()
data_set1 = dataset.iloc[0:3,:]
data_set2 = dataset.iloc[3:6,:]
data_set3 = dataset.iloc[6:9,:]
data_set4 = dataset.iloc[9:12,:]

print data_set1
print data_set2

new_dataset = DM.create_instances_mean([data_set1, data_set2])
new_dataset = DM.create_instances_normal_distribution([data_set1, data_set2])
distance = DM.p_distance(data_set1, data_set2)

print distance

DM = PersonDistanceMetricsOrdering()
data_set1 = dataset.iloc[0:3,:]
data_set2 = dataset.iloc[3:6,:]

distance = DM.euclidean_distance(data_set1, data_set2)
print distance
distance = DM.lag_correlation(data_set1, data_set2, 1)
print distance
distance = DM.dynamic_time_warping(data_set1, data_set2)
print distance

clusteringNH = NonHierarchicalClustering()
dataset1 = clusteringNH.k_means_over_instances(copy.deepcopy(dataset), ['phone_acc_x_kalman', 'phone_acc_y_kalman', 'phone_acc_z_kalman'], 5, 'default', 100, 5)
dataset2 = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), ['phone_acc_x_kalman', 'phone_acc_y_kalman', 'phone_acc_z_kalman'], 5, 'euclidean', 10)
DataViz.plot_clusters_3d(dataset1, ['phone_acc_x_kalman', 'phone_acc_y_kalman', 'phone_acc_z_kalman'], 'cluster', ['label'])
DataViz.plot_silhouette(dataset1, 'cluster', 'silhouette')
print dataset

clustering = NonHierarchicalClustering()
dataset3 = clustering.k_means_over_datasets([data_set1, data_set2, data_set3, data_set4], ['phone_acc_x_kalman', 'phone_acc_y_kalman', 'phone_acc_z_kalman'], 2, clusteringNH.abstraction_normal, 'default', 100, 5)
dataset4 = clustering.k_medoids_over_datasets([data_set1, data_set2, data_set3, data_set4], ['phone_acc_x_kalman', 'phone_acc_y_kalman', 'phone_acc_z_kalman'], 2, clusteringNH.abstraction_euclidean, 'default', 100)
print dataset4

clusteringH = HierarchicalClustering()
dataset5, l = clusteringH.agglomerative_over_instances(copy.deepcopy(dataset), ['phone_acc_x_kalman', 'phone_acc_y_kalman', 'phone_acc_z_kalman'], 5, clusteringNH.euclidean)
DataViz.plot_dendrogram(dataset5, l)
DataViz.plot_clusters_3d(dataset5, ['phone_acc_x_kalman', 'phone_acc_y_kalman', 'phone_acc_z_kalman'], 'cluster', ['label'])
dataset6, l = clusteringH.agglomerative_over_datasets([data_set1, data_set2, data_set3, data_set4], ['phone_acc_x_kalman', 'phone_acc_y_kalman', 'phone_acc_z_kalman'], 2, clusteringNH.abstraction_normal, 'default')
print dataset6