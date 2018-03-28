##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   Chapter 4 - Feature Engineering based on Sensory Data
#   ./crowdsignals_ch4.R
# 
##########################################################################

rm(list=ls())
require(data.table)
require(bit64)
require(plotly)
require(reshape2)
require(zoo) # used for time series representation
source("./Chapter2/createDataset.R")
source("./Chapter2/visiualizeData.R")
source("./chapter4/temporalAbstraction.R")
source("./chapter4/frequencyAbstraction.R")


### settings and data import
resultPath = "./intermediate_datafiles/"
load(file=paste(resultPath,"chapter3_result_final.RData",sep=""))

# compute the number of milliseconds covered by an instane based on the first two rows
timeWindow = as.numeric(difftime(crowdSignalData$time[2],crowdSignalData$time[1],units = "secs"))*1000

# set different window sizes to the number of instances
windowSizes = c(as.integer(5000/timeWindow), as.integer(30000/timeWindow), as.integer(300000/timeWindow))

dataSetCopy = crowdSignalData
for (ws in windowSizes) {
  dataSetCopy = abstractNumerical(dataSetCopy,"acc_phone_x",windowSize = ws,aggregationFunction = "mean")
  dataSetCopy = abstractNumerical(dataSetCopy,"acc_phone_x",windowSize = ws,aggregationFunction = "std")
}

temp = plotDataset(dataSetCopy, c("acc_phone_x", "acc_phone_x_temp_mean", "acc_phone_x_temp_std", "label"),c("exact", "like", "like", "like"))
linePlot = subplot(temp,nrows = length(temp),shareX = TRUE)
print(linePlot)

# add new columns to our dataset
ws = as.integer(30000/timeWindow)

selectedPredictorCols = names(crowdSignalData)
selectedPredictorCols = selectedPredictorCols[!grepl("label",selectedPredictorCols)&!grepl("time",selectedPredictorCols)]

crowdSignalData = abstractNumerical(crowdSignalData,selectedPredictorCols,windowSize = ws,aggregationFunction = "mean")
crowdSignalData = abstractNumerical(crowdSignalData,selectedPredictorCols,windowSize = ws,aggregationFunction = "std")


# select frequent patterns
# crowdSignalData = abstractCategorical(crowdSignalData, c("label"), c("like"), 0.03, as.integer(300000/timeWindow), 2)

### features from the frequency domain
fs = 1000/timeWindow
windowSize = 10000/timeWindow

selectedPredictorCols = c("acc_phone_x","acc_phone_y","acc_phone_z","acc_watch_x","acc_watch_y","acc_watch_z","gyr_phone_x","gyr_phone_y",
                           "gyr_phone_z","gyr_watch_x","gyr_watch_y","gyr_watch_z","mag_phone_x","mag_phone_y","mag_phone_z",
                           "mag_watch_x","mag_watch_y","mag_watch_z")

# accept only a certain percentage of overlap in the windows, otherwise the training examples 
# will be too much alike. 
windowOverlap = 0.9
skipPoints = max(1,as.integer((1-windowOverlap) * ws))
instances2Analyze = seq(1,nrow(crowdSignalData),skipPoints)

crowdSignalData = abstractFrequency(crowdSignalData, selectedPredictorCols, windowSize, fs, instances2Analyze)
crowdSignalData = crowdSignalData[instances2Analyze,]

# temp = plotDataset(dataSetCopy, c("acc_phone_x_max_freq", "acc_phone_x_freq_weighted", "acc_phone_x_pse", "labelOnTable"),c("like", "like", "like", "exact"))
# linePlot = subplot(temp,nrows = length(temp),shareX = TRUE) 
# print(linePlot)


# write output
resultPath = "./intermediate_datafiles/"
save(crowdSignalData, file=paste(resultPath,"chapter4_result.RData",sep=""))
