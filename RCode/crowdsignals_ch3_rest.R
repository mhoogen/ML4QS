##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   Chapter 3 - Handling Noise and Missing Values in Sensory Data
#   ./crowdsignals_ch3_rest.R
# 
##########################################################################

# Import libraries and source utils
rm(list=ls())
require(plotly)
require(reshape2)
require(zoo) # used for time series representation
require(KFAS) # Kalman filtering and smoothing
require(Rlof) # package for local outlier factor
require(signal) # signal processing package (for Butterworth filter)
source("./chapter2/visiualizeData.R")
source("./chapter3/imputationMissingValues.R")
source("./chapter3/kalmanFilter.R")
source("./chapter3/lowPassFilter.R")
source("./chapter3/principalComponent.R")

# Read the result from the previous chapter, and make sture the index is of the type datetime
resultPath = "./intermediate_datafiles/"
load(file=paste(resultPath,"chapter2_result.RData",sep=""))
chapter2Dataset = crowdSignalData

load(file=paste(resultPath,"chapter3_result_outliers.RData",sep=""))

# Compute the number of milliseconds covered by an instane based on the first two rows
timeWindow = as.numeric(difftime(crowdSignalData$time[2],crowdSignalData$time[1],units = "secs"))*1000

### Missing Values
imputedMeanDataset = imputeMean(crowdSignalData, "hr_watch_rate")
imputedMedianDataset = imputeMedian(crowdSignalData, "hr_watch_rate")
imputedInterpolationDataset = imputeInterpolation(crowdSignalData, "hr_watch_rate")
plotImputedValues(crowdSignalData, "hr_watch_rate", list(imputedMeanDataset["hr_watch_rate"], 
  imputedMedianDataset["hr_watch_rate"], imputedInterpolationDataset["hr_watch_rate"]))
rm(imputedMeanDataset,imputedMedianDataset, imputedInterpolationDataset)

# Impute for all columns except for the label in the selected way (interpolation)
col2Inspect = names(crowdSignalData) 
col2Inspect = subset(col2Inspect, (!grepl("label",col2Inspect)&col2Inspect!="time"))
for (col in col2Inspect) 
  crowdSignalData = imputeInterpolation(crowdSignalData, col)

### Kalman filtering on the acc_phone_x attribute
kalmanDataset = applyKalmanFilter(chapter2Dataset , "acc_phone_x")
plot(kalmanDataset$time, kalmanDataset$acc_phone_x_kalman - kalmanDataset$acc_phone_x,type="l", col="blue", ylab = "Difference after Kalman Filter", xlab = "time")
grid()

rm(kalmanDataset) # do not use Kalman filtering

### Low pass filtering: "reduce" signals from the data with more than 1.5 Hz

fs = 1000/timeWindow # sampling frequency [Hz]
cutoff = 1.5 # cut off [Hz]

newDataset = lowPassFilter(crowdSignalData,"acc_phone_x",fs, cutoff, nOrder = 1)
start = round(0.4*nrow(newDataset))
end = round(0.43*nrow(newDataset))
temp = plotDataset(newDataset[start:end,], c("acc_phone_x","acc_phone_x_lowpass"),c("exact","exact"))
subplot(temp,nrows = length(temp),shareX = TRUE)

rm(newDataset)

# Apply lowpass filter to periodic measurements
periodicMeasurements = c("acc_phone_x", "acc_phone_y", "acc_phone_z", "acc_watch_x", "acc_watch_y", "acc_watch_z", "gyr_phone_x", "gyr_phone_y",
                         "gyr_phone_z", "gyr_watch_x", "gyr_watch_y", "gyr_watch_z", "mag_phone_x", "mag_phone_y", "mag_phone_z", "mag_watch_x",
                         "mag_watch_y", "mag_watch_z")

for (col in periodicMeasurements) {
  crowdSignalData = lowPassFilter(crowdSignalData, col, fs, cutoff, nOrder = 1)
  crowdSignalData[,col] = crowdSignalData[,paste(col,"_lowpass",sep="")]
  
}
crowdSignalData = crowdSignalData[,!grepl("_lowpass",colnames(crowdSignalData))]

### Principal Component Analysis (PCA)
col2Inspect = names(crowdSignalData) 
col2Inspect = subset(col2Inspect, (!grepl("label",col2Inspect) & col2Inspect!="time" &
                                     col2Inspect!="hr_watch_rate"))
plot(pcaExplainedVariance(crowdSignalData,col2Inspect),type="b",col="blue",
     xlab = "Number of PC", ylab = "Explained variance")
grid()

crowdSignalData = applyPCA(crowdSignalData, col2Inspect, 7)
temp = plotDataset(crowdSignalData, c("acc_", "gyr_", "hr_watch_rate", "light_phone_lux", "mag_", "press_phone_", "pca_", "label"),NULL)
if (is.null(linePlot)) linePlot = subplot(temp,nrows = length(temp),shareX = TRUE) else
  linePlot = list(linePlot,subplot(temp,nrows = length(temp),shareX = TRUE))

# Write output
resultPath = "./intermediate_datafiles/"
save(crowdSignalData, file=paste(resultPath,"chapter3_result_final.RData",sep=""))
