##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   Chapter 4 - Feature Engineering based on Sensory Data
#   ./chapter4.R
# 
##########################################################################

rm(list=ls())
require(data.table)
require(bit64)
require(plotly)
require(reshape2)
require(stringr)
require(zoo) # used for time series representation
require(tm)
require(SnowballC)
require(magrittr)
require(topicmodels)
source("./Chapter2/createDataset.R")
source("./Chapter2/visiualizeData.R")
source("./Chapter4/textAbstraction.R")
source("./chapter4/ffTransformation.R")


### Settings and data import
resultPath = "./intermediate_datafiles/"
load(file=paste(resultPath,"chapter2_result.RData",sep=""))
chapter2Dataset = crowdSignalData

# Compute the number of milliseconds covered by an instane based on the first two rows
timeWindow = as.numeric(difftime(crowdSignalData$time[2],crowdSignalData$time[1],units = "secs"))*1000


dataset1 = addNumericalDataset(NULL, "merged_accelerometer_1466120659000000000_1466125720000000000.csv","timestamps", c("x","y","z"), "avg", "acc_phone_", datasetPath,timeWindow)

# TOBEDONE: timestamps start and end are always identical, therefore 
dataset1 = addEventDataset(dataset1, "merged_apps_1466120659731000000_1466125749000000000.csv", "start", "end", "app", "binary", datasetPath)

### Text Abstraction
text_example = data.frame(text = as.character(c('getting ready to hit the gym',
                                   'having trouble getting off the couch',
                                   'walking to the gym, it is gonna be a great workout, I feel it',
                                   'the gym did not do it for me, running home',
                                   'still have energy, on my bike now')))

# generates a Corpus using the package tm
vc = generateCorpus(text_example, col = "text", n = 1)
sapply(1:length(vc),function(x){vc[[x]]$content})

# TF IDF matrix is calculated based on words, to change this to n-grams see: http://tm.r-forge.r-project.org/faq.html
tf_idf = TermDocumentMatrix(vc)
inspect(tf_idf)

# LDA topic modeling using the package topicmodels
numOfTopics = 3
topMod = LDA(tf_idf,k=numOfTopics) # k = number of topics
topics(topMod,numOfTopics)

### Numerical data
abstractNumerical(dataset1,"acc_phone_x",5, "slope")

### Fast Fourier Transformation
ffTransformation(crowdSignalData,"acc_phone_x",plot=TRUE)
df=removeComponents(crowdSignalData,"acc_phone_x",components = 300:900)
df1=removeComponents(crowdSignalData,"acc_phone_x",components = 1)

plot(crowdSignalData$time,crowdSignalData$acc_phone_x,col="blue",type="l",xlab = "time",ylab="acc_phone_x")
lines(df$time,df$acc_phone_x,col="red",type="l")
lines(df1$time,df1$acc_phone_x,col="green",type="l")

### Categorical data: TOBEDONE, not implemented due to data problems, see above
