##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   Chapter 7 - Classification
#   ./crowdsignals_ch7_classification.R
# 
##########################################################################

rm(list=ls())
require(mlr) # used for elaborated mechanisms of feature selection
source("./chapter7/prepareDatasetForLearning.R")


### settings and data import
resultPath = "./intermediate_datafiles/"
load(file=paste(resultPath,"chapter5_result.RData",sep=""))

# We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
# for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
# cases where we do not know the label.
res = splitSingleDatasetClassification(df=crowdSignalData,classLabels = "label", matching = "like", trainingFrac = 0.7)
trainX = res$trainingSetX
trainY = res$trainingSety
testX = res$testSetX
testY = res$testSety


print(paste("Training set length is: ", nrow(trainX)))
print(paste("Test set length is: ", nrow(testX)))

# select subsets of the features that we will consider:
basicFeatures = c("acc_phone_x","acc_phone_y","acc_phone_z","acc_watch_x","acc_watch_y","acc_watch_z","gyr_phone_x","gyr_phone_y","gyr_phone_z","gyr_watch_x","gyr_watch_y","gyr_watch_z",
                  "hr_watch_rate", "light_phone_lux","mag_phone_x","mag_phone_y","mag_phone_z","mag_watch_x","mag_watch_y","mag_watch_z","press_phone_pressure")
pcaFeatures = c("pca_1","pca_2","pca_3","pca_4","pca_5","pca_6","pca_7")
timeFeatures = names(crowdSignalData)[grepl("_temp_",names(crowdSignalData))]
freqFeatures = names(crowdSignalData)[grepl("_freq",names(crowdSignalData))]
clusterFeatures = "cluster"

print(paste("#basic features: ", length(basicFeatures)))
print(paste("#PCA features: ", length(pcaFeatures)))
print(paste("#time features: ", length(timeFeatures)))
print(paste("#frequency features: ", length(freqFeatures)))
print(paste("#cluster features: ", length(clusterFeatures)))

featuresUptoChapter3 = c(basicFeatures,pcaFeatures)
featuresUptoChapter4 = c(featuresUptoChapter3, timeFeatures, freqFeatures)
featuresUptoChapter5 = c(featuresUptoChapter4, clusterFeatures)

### determine relevant features by forward selection on a tree learner
# we use the mlr/mbench packages in R to do forward selection elegantly
# source: https://mlr-org.github.io/mlr-tutorial/release/html/feature_selection/index.html
trainingSet = trainX[,featuresUptoChapter5]
trainingSet$class = as.factor(trainY)

# generate classification task
classTask = makeClassifTask(data = trainingSet, target="class")

# specify search strategy (forward selection) 
ctrl = makeFeatSelControlSequential(method = "sfs",  alpha =0, max.features = 50)
rdesc = makeResampleDesc("CV", iters = 2)
selfeats = selectFeatures(learner = "classif.rpart", task = classTask, resampling = rdesc, control = ctrl,
                        measure = acc, show.info = TRUE)
print(paste("Selected features",selfeats$x))


### explore regularization (use glmnet)
trainingSet = trainX[,featuresUptoChapter4]#[,selectedFeatures]
trainingSet$class = as.factor(trainY)
testSet = testX[,featuresUptoChapter4]#[,selectedFeatures]
testSet$class = as.factor(testY)

classTask = makeClassifTask(data = trainingSet, target="class")

#learner = makeLearner("classif.mlp", size = 250, maxit = 500)#, decay = 1, size = 50, maxit = 1000)

regParameters = c( 0.0001, 0.001, 0.01, 0.1, 1, 10)
performanceTraining = c()
performanceTest = c()
k = 0

for (lambda in regParameters) {
  k = k + 1
  learner = makeLearner("classif.glmnet", s = lambda)
  model = train(learner, classTask)
  pred = predict(model, task = classTask)
  performanceTraining = c(performanceTraining,
                          performance(pred, measures = list(acc)))
  
  pred = predict(model, makeClassifTask(data=testSet,target="class"))
  performanceTest = c(performanceTest,
                          performance(pred, measures = list(acc)))
}

plot(regParameters,performanceTraining,log = "x",ylim=c(0,1.05),type="l",ylab="Accuracy")
lines(regParameters,performanceTest,lty=2,col="red")
grid()

# selected features (see python code)
selectedFeatures = c("acc_phone_y_freq_0_Hz_ws_40", "press_phone_pressure_temp_mean_ws_120", "gyr_phone_x_temp_std_ws_120",
                     "mag_watch_y_pse", "mag_phone_z_max_freq", "gyr_watch_y_freq_weighted", "gyr_phone_y_freq_1_Hz_ws_40",
                     "acc_phone_x_freq_1.9_Hz_ws_40", "mag_watch_z_freq_0.9_Hz_ws_40", "acc_watch_y_freq_0.5_Hz_ws_40")
