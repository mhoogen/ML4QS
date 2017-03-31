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
import copy
import pandas as pd

# Of course we repeat some stuff from Chapter 3, namely to load the dataset

DataViz = VisualizeDataset()
milliseconds_per_instance = 5000

# Create an initial dataset object
DataSet = CreateDataset('/Users/markhoogendoorn/Dropbox/Quantified-Self-Book/datasets/crowdsignals.io/Csv-merged/', milliseconds_per_instance)
#
#DataSet.add_numerical_dataset('merged_accelerometer_1466120659000000000_1466125720000000000.csv', 'timestamps', ['x','y','z'], 'avg', 'phone_acc_')
DataSet.add_event_dataset('merged_apps_1466120659731000000_1466125749000000000.csv', 'start', 'end', 'app', 'binary')
#DataSet.add_numerical_dataset('merged_msBandSkinTemperature_1466120660000000000_1466125634000000000.csv', 'timestamps', ['temperature'], 'avg', 'watch_skin_')

# Chapter 5: Identifying aggregate attributes.

#NumAbs = NumericalAbstraction()
dataset = DataSet.data_table
#dataset = NumAbs.abstract_numerical(dataset, ['phone_acc_x'], 5, 'slope')
#dataset = NumAbs.abstract_numerical(dataset, ['phone_acc_x'], 5, 'mean')
#DataViz.plot_dataset(dataset, ['phone_acc_x', 'phone_acc_x_temp_slope', 'phone_acc_x_temp_mean'], ['exact', 'exact', 'exact'], ['line', 'line', 'line'])

CatAbs = CategoricalAbstraction()
#dataset = CatAbs.abstract_categorical(dataset, ['app'], ['like'], 0.01, 5, 2)
#print len(dataset.columns)
#print dataset

# Some examples for text processing....

text_example = pd.DataFrame({'text': pd.Series(['getting ready to hit the gym','having trouble getting off the couch',
                                             'walking to the gym, it is gonna be a great workout, I feel it',
                                             'the gym did not do it for me, running home','still have energy, on my bike now'])})

#TA = TextAbstraction()
#dataset = TA.bag_of_words(text_example, ['text'], 2)
#dataset = TA.tf_idf(text_example, ['text'])
#dataset = TA.topic_modeling(text_example, ['text'], 3)
#print dataset