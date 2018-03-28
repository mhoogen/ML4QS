##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   Chapter 7 - Classification
#   ./chapter7/prepareDatasetForLearning.R
# 
##########################################################################

# This function creates a single class column based on a set of binary class columns.
# it essentially merges them. It removes the old label columns.
assignLabel = function(df, classLabels){
  # find which columns are relevant based on the possibly partial classLabel
  # specification.
  labels = names(df)[startsWith(names(df),classLabels)]
  temp = df[,labels]
  df$class = apply(temp,1,which.max)*apply(temp,1,sum)
  df$class = ifelse(df$class>0,labels[df$class],NA)
  return(df)  
}

# Split a dataset of a single person for a classificaiton problem with the specified class columns classLabels.
# We can have multiple targets if we want. It assumes a list in 'classLabels'
# If 'like' is specified in matching, we will merge the columns that contain the classLabels into a single
# columns. We can select a filter for rows where we are unable to identifty a unique
# class and we can select whether we have a temporal dataset or not. In the former, we will select the first
# trainingFrac of the data for training and the last 1-trainingFrac for testing. Otherwise, we select points randomly.
# We return a training set, the labels of the training set, and the same for a test set. We can set the random seed
# to make the split reproducible.
splitSingleDatasetClassification = function(df, classLabels, matching, trainingFrac, filter=TRUE, temporal = FALSE, randomState = 0) {
  
  # features are the ones not in the class label (include timestamp, or use & names(df)!="time")
  features = names(df)[!startsWith(names(df),classLabels)]
  
  # create a single class column if we have the 'like' option.
  if (matching == "like") {
    df = assignLabel(df, classLabels)
    classLabels = "class"
  } 
  
  # filter NA is desired and those for which we cannot determine the class should be removed.
  if(filter) df = subset(df,!class==0) #[!is.na(df$class),]
  
  # For temporal data, we select the desired fraction of training data from the first part
  # and use the rest as test set.
  if (temporal) {
    endTrainingSet = int(trainingFrac * nrow(df))
    trainingSetX = df[1:endTrainingSet,features]
    trainingSety = df[1:endTrainingSet,classLabels]
    testSetX = df[(endTrainingSet+1):nrow(df),features]
    testSety = df[(endTrainingSet+1):nrow(df),classLabels]
  } else {
    # For non temporal data we use a standard function to randomly split the dataset.
    trainTestSplit = sample(nrow(df),trainingFrac*nrow(df))
    trainingSetX = df[trainTestSplit,features]
    trainingSety = df[trainTestSplit,classLabels]
    testSetX = df[-trainTestSplit,features]
    testSety = df[-trainTestSplit,classLabels]
  }
  return(list(trainingSetX = trainingSetX, trainingSety = trainingSety,
              testSetX = testSetX, testSety = testSety))
}