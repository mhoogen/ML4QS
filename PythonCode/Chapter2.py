from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import FourierTransformation

# Chapter 3: Initial exploration of the dataset.

milliseconds_per_instance = 1000

# Create an initial dataset object
DataSet = CreateDataset('/Users/markhoogendoorn/Dropbox/Quantified-Self-Book/datasets/crowdsignals.io/Csv-merged/', milliseconds_per_instance)

# Add some of our measurements to it.
DataSet.add_numerical_dataset('merged_accelerometer_1466120659000000000_1466125720000000000.csv', 'timestamps', ['x','y','z'], 'avg', 'phone_acc_')
DataSet.add_numerical_dataset('merged_msBandAccelerometer_1466120661000000000_1466125729000000000.csv', 'timestamps', ['x','y','z'], 'avg', 'watch_acc_')
DataSet.add_event_dataset('merged_interval_label_1466120784502000000_1466125692364000000.csv', 'start', 'end', 'label', 'binary')
DataSet.add_numerical_dataset('merged_msBandDistance_1466120662000000000_1466125737000000000.csv', 'timestamps', ['speed'], 'avg', 'watch_dist_')
DataSet.add_numerical_dataset('merged_msBandSkinTemperature_1466120660000000000_1466125634000000000.csv', 'timestamps', ['temperature'], 'avg', 'watch_skin_')
DataSet.add_numerical_dataset('merged_msBandAmbientLight_1466120661000000000_1466125714000000000.csv', 'timestamps', ['lux'], 'avg', 'watch_light_')

# And plot some interesting things:

dataset = DataSet.data_table

DataViz = VisualizeDataset()
DataViz.plot_dataset(dataset, ['phone_acc_', 'watch_acc_', 'watch_dist_', 'label', 'watch_skin_', 'watch_light_'], ['like', 'like', 'like', 'like', 'like','like',], ['line', 'line', 'line', 'points', 'line', 'line'])
#DataViz.plot_dataset(dataset, ['phone_acc_'], ['line'])
