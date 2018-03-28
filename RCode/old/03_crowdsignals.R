rm(list=ls())
require(data.table)
require(bit64)
require(plotly)
require(reshape2)
source("./Chapter3/createDataset.R")
source("./Chapter3/visiualizeData.R")

# Settings
datasetPath = '../datasets/crowdsignals.io/csv-participant-one/'
timeWindow = 60000

# We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
# and aggregate the values per timestep by averaging the values/
dataset1 = addNumericalDataset(NULL, "accelerometer_phone.csv","timestamps", c("x","y","z"), "avg", "acc_phone_", datasetPath,timeWindow)
dataset1 = addNumericalDataset(dataset1, "accelerometer_smartwatch.csv","timestamps", c("x","y","z"), "avg", "acc_watch_", datasetPath,timeWindow)

# We add the gyroscope data (continuous numerical measurements) of the phone and the smartwatch
# and aggregate the values per timestep by averaging the values
dataset1 = addNumericalDataset(dataset1, "gyroscope_phone.csv","timestamps", c("x","y","z"), "avg", "gyr_phone_", datasetPath,timeWindow)
dataset1 = addNumericalDataset(dataset1, "gyroscope_smartwatch.csv","timestamps", c("x","y","z"), "avg", "gyr_watch_", datasetPath,timeWindow)

# We add the heart rate (continuous numerical measurements) and aggregate by averaging again
dataset1 = addNumericalDataset(dataset1, "heart_rate_smartwatch.csv","timestamps", c("rate"), "avg", "hr_watch_", datasetPath,timeWindow)


# We add the labels provided by the users. These are categorical events that might overlap. We add them
# as binary attributes (i.e. ad a one to the attribute representing the specific value for the label if it
# occurs within an interval).
dataset1 = addEventDataset(dataset1, "labels.csv", "label_start", "label_end", "label", "binary", datasetPath)
  

# We add the amount of light sensed by the phone (continuous numerical measurements) and aggregate by averaging again
dataset1 = addNumericalDataset(dataset1, "light_phone.csv","timestamps", c("lux"), "avg", "light_phone_", datasetPath,timeWindow)


# We add the magnetometer data (continuous numerical measurements) of the phone and the smartwatch
# and aggregate the values per timestep by averaging the values/
dataset1 = addNumericalDataset(dataset1, "magnetometer_phone.csv","timestamps", c("x","y","z"), "avg", "mag_phone_", datasetPath,timeWindow)
dataset1 = addNumericalDataset(dataset1, "magnetometer_smartwatch.csv","timestamps", c("x","y","z"), "avg", "mag_watch_", datasetPath,timeWindow)

# We add the pressure sensed by the phone (continuous numerical measurements) and aggregate by averaging again
dataset1 = addNumericalDataset(dataset1, "pressure_phone.csv","timestamps", c("pressure"), "avg", "press_phone_", datasetPath,timeWindow)
dataset1$time = as.POSIXct(dataset1$time/1000, origin="1970-01-01")

p = plotDataset(dataset1, c('acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_','label'),NULL,NULL)
subplot(p,nrows = length(p),shareX = TRUE)
