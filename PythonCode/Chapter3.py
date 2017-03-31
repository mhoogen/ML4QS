from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import FourierTransformation
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters
import copy

# Of course we repeat some stuff from Chapter 3, namely to load the dataset

DataViz = VisualizeDataset()
milliseconds_per_instance = 5000

# Create an initial dataset object
DataSet = CreateDataset('/Users/markhoogendoorn/Dropbox/Quantified-Self-Book/datasets/crowdsignals.io/Csv-merged/', milliseconds_per_instance)

# Add some of our measurements to it.
DataSet.add_numerical_dataset('merged_accelerometer_1466120659000000000_1466125720000000000.csv', 'timestamps', ['x','y','z'], 'avg', 'phone_acc_')
#DataSet.add_numerical_dataset('merged_msBandAccelerometer_1466120661000000000_1466125729000000000.csv', 'timestamps', ['x','y','z'], 'avg', 'watch_acc_')
#DataSet.add_event_dataset('merged_interval_label_1466120784502000000_1466125692364000000.csv', 'start', 'end', 'label', 'binary')
#DataSet.add_numerical_dataset('merged_msBandDistance_1466120662000000000_1466125737000000000.csv', 'timestamps', ['speed'], 'avg', 'watch_dist_')
DataSet.add_numerical_dataset('merged_msBandSkinTemperature_1466120660000000000_1466125634000000000.csv', 'timestamps', ['temperature'], 'avg', 'watch_skin_')
#DataSet.add_numerical_dataset('merged_msBandAmbientLight_1466120661000000000_1466125714000000000.csv', 'timestamps', ['lux'], 'avg', 'watch_light_')

# Chapter 4: Handling Noise. Let us try for the X-axis of our Phone accelerometer

cols = DataSet.get_relevant_columns(['phone_acc_', 'watch_acc', 'watch_dist_', 'watch_light_'])
dataset = DataSet.data_table

# Fourier stuff

# DataTransformX = FourierTransformation()
# DataTransformY = FourierTransformation()
# DataTransformZ = FourierTransformation()
# samples = len(dataset.index)
# freq, ampl_real, ampl_imag = DataTransformX.find_fft_transformation(dataset, 'phone_acc_x', samples, float(1000)/milliseconds_per_instance)
# DataViz.plot_fourier_amplitudes(freq, ampl_real, ampl_imag)
# DataTransformY.find_fft_transformation(dataset, 'phone_acc_y', samples, float(1000)/milliseconds_per_instance)
# DataTransformZ.find_fft_transformation(dataset, 'phone_acc_z', samples, float(1000)/milliseconds_per_instance)

# dataset = DataTransformX.remove_components(dataset, 'phone_acc_x', range(100, samples))
# dataset = DataTransformY.remove_components(dataset, 'phone_acc_y', range(100, samples))
# dataset = DataTransformZ.remove_components(dataset, 'phone_acc_z', range(100, samples))
# DataViz.plot_dataset(dataset, ['phone_acc_x', 'phone_acc_y', 'phone_acc_z'], ['like', 'like', 'like'], ['line', 'line', 'line'])

# PCA

#PCA = PrincipalComponentAnalysis()
#pc_values = PCA.determine_pc_explained_variance(dataset, cols)

# Look at the variance explained
#print pc_values

#dataset = PCA.apply_pca(copy.deepcopy(dataset), cols, 2)

#DataViz.plot_dataset(dataset, ['pca_', 'label'], ['like', 'like'], ['line', 'points'])

# Outlier

#OutlierDistr = DistributionBasedOutlierDetection()
# dataset = OutlierDistr.chauvenet(dataset, 'phone_acc_x')

# DataViz.plot_binary_outliers(dataset, 'phone_acc_x')

#dataset = OutlierDistr.mixture_model(dataset, 'phone_acc_x')
#DataViz.plot_dataset(dataset, ['phone_acc_x', 'phone_acc_x_mixture'], ['exact','exact'], ['line', 'points'])

#OutlierDist = DistanceBasedOutlierDetection()
#dataset = OutlierDist.simple_distance_based(dataset, ['phone_acc_x'], 'euclidean', 0.8, 0.005)
#DataViz.plot_binary_outliers(dataset, 'phone_acc_x', 'simple_dist_outlier')

#dataset = OutlierDist.local_outlier_factor(dataset, ['phone_acc_x'], 'euclidean', 5)
#DataViz.plot_dataset(dataset, ['phone_acc_x', 'lof'], ['exact','exact'], ['line', 'points'])

# Missing Value imputation

#MisVal = ImputationMissingValues()
#imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'watch_skin_temperature')
#imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'watch_skin_temperature')
#imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'watch_skin_temperature')
#DataViz.plot_imputed_values(dataset, 'watch_skin_temperature', imputed_mean_dataset['watch_skin_temperature'], imputed_median_dataset['watch_skin_temperature'], imputed_interpolation_dataset['watch_skin_temperature'])

# Kalman Filter

KalFilter = KalmanFilters()
dataset = KalFilter.apply_kalman_filter(dataset, 'watch_skin_temperature')
DataViz.plot_imputed_values(dataset, 'watch_skin_temperature', dataset['watch_skin_temperature_kalman'])
# DataViz.plot_dataset(dataset, ['watch_skin_temperature', 'watch_skin_temperature_kalman'], ['exact','exact'], ['line', 'line'])