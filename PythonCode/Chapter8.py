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
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.FeatureSelection import FeatureSelectionRegression
from Chapter8.LearningAlgorithmsTemporal import TemporalClassificationAlgorithms
from Chapter8.LearningAlgorithmsTemporal import TemporalRegressionAlgorithms
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
data_set1 = pd.DataFrame(dataset.iloc[0:3,:])
data_set2 = pd.DataFrame(dataset.iloc[3:6,:])
data_set3 = pd.DataFrame(dataset.iloc[6:9,:])
data_set4 = pd.DataFrame(dataset.iloc[9:12,:])


# Try to computer the distance between two data elements

prepare = PrepareDatasetForLearning()
train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(copy.deepcopy(dataset), ['label'], 'like', 0.9, filter=True, temporal=True)
#train_X, test_X, train_y, test_y = prepare.split_multiple_datasets_classification([copy.deepcopy(dataset), copy.deepcopy(dataset)], ['label'], 'like', 0.5, filter=False, temporal=True, unknown_users=False)

tc = TemporalClassificationAlgorithms()

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = tc.reservoir_computing(train_X, train_y, test_X, test_y, 5, per_time_step=True)
class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = tc.recurrent_neural_network(train_X, train_y, test_X, test_y, 10, iterations=100)
print class_train_y
print class_train_prob_y
print class_test_y
print class_test_prob_y
eval = ClassificationEvaluation()

train_acc = eval.accuracy(train_y, class_train_y)
test_acc = eval.accuracy(test_y, class_test_y)
train_p = eval.precision(train_y, class_train_y)
test_p = eval.precision(test_y, class_test_y)
print train_p, test_p
train_r = eval.recall(train_y, class_train_y)
test_r = eval.recall(test_y, class_test_y)
print train_r, test_r
train_f1 = eval.f1(train_y, class_train_y)
test_f1 = eval.f1(test_y, class_test_y)
print train_f1, test_f1
train_cm = eval.confusion_matrix(train_y, class_train_y, class_train_prob_y.columns)
test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)
print train_cm
print test_cm

DataViz.plot_confusion_matrix(train_cm, class_train_prob_y.columns, normalize=False)

train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression(copy.deepcopy(dataset), ['phone_acc_x_kalman', 'phone_acc_y_kalman'], 0.9, filter=True, temporal=True)
#train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression(dataset, ['phone_acc_x_kalman'], 0.9, filter=True, temporal=True)
#train_X, test_X, train_y, test_y = prepare.split_multiple_datasets_regression([copy.deepcopy(dataset), copy.deepcopy(dataset)], 'phone_acc_x_kalman', 0.5, filter=False, temporal=True, unknown_users=True)
fs = FeatureSelectionRegression()
features, feature_order, feature_scores = fs.forward_selection(2, train_X, train_y)
features = fs.backward_selection(2, train_X, train_y)
features, feature_scores = fs.pearson_selection(2, train_X, train_y['phone_acc_x_kalman'])
print features

learner = TemporalRegressionAlgorithms()
output_sets = learner.dynamical_systems_model_nsga_2(train_X, train_y, test_X, test_y, ['self.phone_acc_x_kalman', 'self.phone_acc_y_kalman', 'self.phone_acc_z_kalman'],
                                                     ['self.a * self.phone_acc_x_kalman + self.b * self.phone_acc_y_kalman', 'self.c * self.phone_acc_y_kalman + self.d * self.phone_acc_z_kalman', 'self.e * self.phone_acc_x_kalman + self.f * self.phone_acc_z_kalman'],
                                                     ['self.phone_acc_x_kalman', 'self.phone_acc_y_kalman'],
                                                     ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                                     pop_size=10, max_generations=10, per_time_step=True)
DataViz.plot_pareto_front(output_sets)

DataViz.plot_numerical_prediction_versus_real_dynsys_mo(train_X.index, train_y, test_X.index, test_y, output_sets, 0, 'phone_acc_y_kalman')

regr_train_y, regr_test_y = learner.dynamical_systems_model_ga(train_X, train_y, test_X, test_y, ['self.phone_acc_x_kalman', 'self.phone_acc_y_kalman', 'self.phone_acc_z_kalman'],
                                                     ['self.a * self.phone_acc_x_kalman + self.b * self.phone_acc_y_kalman', 'self.c * self.phone_acc_y_kalman + self.d * self.phone_acc_z_kalman', 'self.e * self.phone_acc_x_kalman + self.f * self.phone_acc_z_kalman'],
                                                     ['self.phone_acc_x_kalman', 'self.phone_acc_y_kalman'],
                                                     ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                                     pop_size=5, max_generations=10, per_time_step=True)

regr_train_y, regr_test_y = learner.dynamical_systems_model_sa(train_X, train_y, test_X, test_y, ['self.phone_acc_x_kalman', 'self.phone_acc_y_kalman', 'self.phone_acc_z_kalman'],
                                                     ['self.a * self.phone_acc_x_kalman + self.b * self.phone_acc_y_kalman', 'self.c * self.phone_acc_y_kalman + self.d * self.phone_acc_z_kalman', 'self.e * self.phone_acc_x_kalman + self.f * self.phone_acc_z_kalman'],
                                                     ['self.phone_acc_x_kalman', 'self.phone_acc_y_kalman'],
                                                     ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                                     max_generations=10, per_time_step=True)
regr_train_y, regr_test_y = learner.reservoir_computing(train_X, train_y, test_X, test_y, 10, per_time_step=True)

regr_train_y, regr_test_y = learner.recurrent_neural_network(train_X, train_y, test_X, test_y, 10, iterations=100)
DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y['phone_acc_x_kalman'], regr_train_y['phone_acc_x_kalman'], test_X.index, test_y['phone_acc_x_kalman'], regr_test_y['phone_acc_x_kalman'], 'phone_acc_x_kalman')


eval = RegressionEvaluation()
train_mse = eval.mean_squared_error(train_y, regr_train_y)
test_mse = eval.mean_squared_error(test_y, regr_test_y)
print train_mse, test_mse
train_mae = eval.mean_absolute_error(train_y, regr_train_y)
test_mae = eval.mean_absolute_error(test_y, regr_test_y)
print train_mae, test_mae

