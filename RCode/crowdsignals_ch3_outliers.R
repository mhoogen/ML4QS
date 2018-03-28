##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   Chapter 3 - Handling Noise and Missing Values in Sensory Data
#   ./crowdsignals_ch3_outliers.R
# 
##########################################################################

# Import libraries and source utils
rm(list=ls())
require(plotly)
require(reshape2)
require(zoo) # used for time series representation
require(KFAS) # Kalman filtering and smoothing
require(Rlof) # package for local outlier factor
source("./chapter2/visiualizeData.R")
source("./chapter3/distributionBasedOutlierDetection.R")
source("./chapter3/distanceBasedOutlierDetection.R")

# Read the result from the previous chapter, and make sture the index is of the type datetime
resultPath = './intermediate_datafiles/'
load(file=paste(resultPath,"chapter2_result.RData",sep=""))

# Compute the number of milliseconds covered by an instane based on the first two rows
timeWindow = as.numeric(difftime(crowdSignalData$time[2],crowdSignalData$time[1],units = "secs"))*1000

# Determine the columns we want to experiment on.
col2Inspect = c('acc_phone_x', 'light_phone_lux')

# Investigate the approaches for all attributes specified in col2Inspect
for (col in col2Inspect) {
  # Note that we have done some optimization of the parameter values for each of the 
  # approaches by visual inspection
  print(paste("====",col,"===="))
  
  # Chauvenet criterion
  print(paste("Chauvenet:",Sys.time()))
  crowdSignalData = chauvenet(crowdSignalData, col)
  plotBinaryOutliers(crowdSignalData,col,paste(col,'_chauvenetOutlier',sep=""))

  # Mixture models
  print(paste("Mixture model:",Sys.time()))
  crowdSignalData = mixtureModel(crowdSignalData,col,numOfComponents = 3, plot = FALSE)
  crowdSignalData[,paste(col,"_mixtureOutlier",sep="")] = crowdSignalData[,paste(col,"_mixtureProb",sep="")]<0.0001
  plotBinaryOutliers(crowdSignalData,col,paste(col,"_mixtureOutlier",sep=""))

  # Distance based approaches require to calculate pair-wise distances, the computation time and
  # the memory requirements increase with N^2, for demonstration purposes we therefore reduce the sample size 
  crowdSignalData = crowdSignalData[1:10000,]
  
  # Simple distance based detection
  print(paste("Simple distance based:",Sys.time()))
  crowdSignalData = simpleDistanceBased(crowdSignalData,col, 0.5, 0.99)
  plotBinaryOutliers(crowdSignalData,col,paste(col,"_simpleDistOutlier",sep=""))
  
  # Local Outlier Factor: we use the Rlof package which implements LOF
  print(paste("Local Outlier Factor:",Sys.time()))
  crowdSignalData = localOutlierFactor(crowdSignalData, col, dFunction = "euclidean", 5)
  crowdSignalData[, paste(col,"_lofOutlier",sep="")] =  crowdSignalData$lof > 5
  plotBinaryOutliers(crowdSignalData, col, paste(col,"_lofOutlier",sep=""))
}

# Reload the result from Chapter 2 and apply Chauvenet criterion to all attributes but the labels
load(file=paste(resultPath,"chapter2_result.RData",sep=""))
col2Inspect = names(crowdSignalData) 
col2Inspect = subset(col2Inspect, (!grepl("label",col2Inspect)&col2Inspect!="time"))
for (col in col2Inspect) {
  
  # Chauvenet criterion
  crowdSignalData = chauvenet(crowdSignalData, col)
  plotBinaryOutliers(crowdSignalData,col,paste(col,'_chauvenetOutlier',sep=""))
  
  # Set outliers to NA
  temp = crowdSignalData[,paste(col,'_chauvenetOutlier',sep="")]
  temp = ifelse(is.na(temp),FALSE,temp)
  crowdSignalData[temp,col] = NA
}

# Delete outlier columns
crowdSignalData = crowdSignalData[,!grepl("_chauvenetOutlier",names(crowdSignalData))]

# Save to file
resultPath = './intermediate_datafiles/'
save(crowdSignalData, file=paste(resultPath,"chapter3_result_outliers.RData",sep=""))