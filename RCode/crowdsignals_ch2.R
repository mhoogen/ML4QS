##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   Chapter 2 - Sensory Data
#   ./crowdsignals_ch2.R
# 
##########################################################################

# Import libraries and source utils
rm(list=ls())
require(data.table)
require(bit64)
require(plotly)
require(reshape2)
source("./chapter2/createDataset.R")
source("./chapter2/visiualizeData.R")

# Set a granularity (i.e. how big are our discrete time steps). We start very
# coarse grained, namely one measurement per minute, and secondly use four measurements
# per second
datasetPath = "../datasets/crowdsignals.io/csv-participant-one/"
#granularities = c(60000,250)
granularities = c(250)

boxPlot = NULL 
linePlot = NULL
for (timeWindow in granularities) {
  
  # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
  # and aggregate the values per timestep by averaging the values/
  crowdSignalData = addNumericalDataset(NULL, "accelerometer_phone.csv","timestamps", c("x","y","z"), "avg", "acc_phone_", datasetPath,timeWindow)
  crowdSignalData = addNumericalDataset(crowdSignalData, "accelerometer_smartwatch.csv","timestamps", c("x","y","z"), "avg", "acc_watch_", datasetPath,timeWindow)
  
  # We add the gyroscope data (continuous numerical measurements) of the phone and the smartwatch
  # and aggregate the values per timestep by averaging the values
  crowdSignalData = addNumericalDataset(crowdSignalData, "gyroscope_phone.csv","timestamps", c("x","y","z"), "avg", "gyr_phone_", datasetPath,timeWindow)
  crowdSignalData = addNumericalDataset(crowdSignalData, "gyroscope_smartwatch.csv","timestamps", c("x","y","z"), "avg", "gyr_watch_", datasetPath,timeWindow)
  
  # We add the heart rate (continuous numerical measurements) and aggregate by averaging again
  crowdSignalData = addNumericalDataset(crowdSignalData, "heart_rate_smartwatch.csv","timestamps", c("rate"), "avg", "hr_watch_", datasetPath,timeWindow)
  
  # We add the labels provided by the users. These are categorical events that might overlap. We add them
  # as binary attributes (i.e. ad a one to the attribute representing the specific value for the label if it
  # occurs within an interval).
  crowdSignalData = addEventDataset(crowdSignalData, "labels.csv", "label_start", "label_end", "label", "binary", datasetPath)
  
  # We add the amount of light sensed by the phone (continuous numerical measurements) and aggregate by averaging again
  crowdSignalData = addNumericalDataset(crowdSignalData, "light_phone.csv","timestamps", c("lux"), "avg", "light_phone_", datasetPath,timeWindow)
  
  # We add the magnetometer data (continuous numerical measurements) of the phone and the smartwatch
  # and aggregate the values per timestep by averaging the values/
  crowdSignalData = addNumericalDataset(crowdSignalData, "magnetometer_phone.csv","timestamps", c("x","y","z"), "avg", "mag_phone_", datasetPath,timeWindow)
  crowdSignalData = addNumericalDataset(crowdSignalData, "magnetometer_smartwatch.csv","timestamps", c("x","y","z"), "avg", "mag_watch_", datasetPath,timeWindow)
  
  # We add the pressure sensed by the phone (continuous numerical measurements) and aggregate by averaging again
  crowdSignalData = addNumericalDataset(crowdSignalData, "pressure_phone.csv","timestamps", c("pressure"), "avg", "press_phone_", datasetPath,timeWindow)
  crowdSignalData$time = as.POSIXct(crowdSignalData$time/1000, origin="1970-01-01")
  
  # Lineplot
  # to plot, simply use print, e.g. "print(p[1])"
  temp = plotDataset(crowdSignalData, c("acc_", "gyr_", "hr_watch_rate", "light_phone_lux", "mag_", "press_phone_","label"),NULL)
  if (is.null(linePlot)) linePlot = subplot(temp,nrows = length(temp),shareX = TRUE) else
    linePlot = list(linePlot,subplot(temp,nrows = length(temp),shareX = TRUE))
  
  # Boxplot
  if(is.null(boxPlot)) boxPlot = list(plotBoxplot(crowdSignalData, c("acc_phone_x","acc_phone_y","acc_phone_z","acc_watch_x","acc_watch_y","acc_watch_z"))) else
    boxPlot = list(boxPlot,plotBoxplot(crowdSignalData, c("acc_phone_x","acc_phone_y","acc_phone_z","acc_watch_x","acc_watch_y","acc_watch_z")))   
}

# Only the dataset with the last value for granularities is saved 
resultPath = "./intermediate_datafiles/"
save(crowdSignalData, file=paste(resultPath,"chapter2_result.RData",sep=""))

# Save as CSV file

# crowdSignalData$time = as.character(crowdSignalData$time,format="%Y-%m-%d %H:%M:%OS4")
# write.csv(x=crowdSignalData, paste(resultPath,"chapter2_result.csv",sep=""))