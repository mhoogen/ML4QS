##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   Chapter 4 - Feature Engineering based on Sensory Data
#   ./chapter4/temporalAbstraction.R
# 
##########################################################################

# This function aggregates a list of values using the specified aggregation
# function (which can be 'mean', 'max', 'min', 'median', 'std', 'slope')
aggregateValue = function(data, aggregationFunction) {
  if (aggregationFunction == "mean") return(mean(data))
  if (aggregationFunction == "max") return(max(data))
  if (aggregationFunction == "min") return(min(data))
  if (aggregationFunction == "median") return(median(data))
  if (aggregationFunction == "std") return(sd(data))
  if (aggregationFunction == "slope") return(as.numeric(lm(data~(x=1:length(data)))$coef[2]))
  stop(paste("aggregationFunction:",aggregationFunction,"not found"))
}


abstractNumerical = function(df, cols, windowSize, aggregationFunction) {
  for (col in cols) {
    newColName = paste(col,"_temp_",aggregationFunction,"_ws_",windowSize,sep = "")
    data = df[,col]
    aggData = sapply((windowSize+1):length(data), function(x) {aggregateValue(data[(x-windowSize):x],aggregationFunction)})
    df[floor(windowSize/2)+(1:(length(data)-windowSize)),newColName] = aggData
  }
  return(df)
}

# Function to abstract categorical data. Note that we assume a list of binary columns representing
# the different categories 
abstractCategorical = function(df, cols, match, minSupport, windowSize, maxPatternSize) {
  # find all the relevant columns of binary attributes.
  colNames = names(df)
  selectedPatterns = c()
  relevantCols = c()
  
  for (i in 1:length(cols)) {
    if(match[i]=="exact") relevantCols = c(relevantCols,cols[i]) else
      relevantCols = c(relevantCols, colNames[grepl(cols[i],colNames)])
  }
  
  # generate the one patterns first
  potential1Patterns = as.list(relevantCols)
  temp = selectKPatterns(df,potential1Patterns,minSupport,windowSize)
  df = temp[[1]]
  onePatterns = temp[[2]]
  selectedPatterns[[length(selectedPatterns)+1]] = onePatterns
  print(paste("Number of patterns of size 1:",length(onePatterns)))

  k = 1
  kPatterns = onePatterns
  while(k< maxPatternSize & length(kPatterns)>0) {
    k = k + 1
    potentialKPatterns = extendKPatterns(kPatterns,onePatterns)
    print(potentialKPatterns)
    temp = selectKPatterns(df, potentialKPatterns, minSupport, windowSize)
    df = temp[[1]]
    kPatterns = temp[[2]]
    selectedPatterns[[length(selectedPatterns)+1]] = kPatterns
    print(paste("Number of patterns of size",k,":",length(kPatterns)))
  }
  return(df)
}

# selects the patterns from 'patterns' that meet the minimum support in the dataset
# given the window size.
selectKPatterns = function(df, patterns, minSupport, windowSize){
  sP = list()
  for (pattern in patterns) {
    # determine the number of occurrences of a pattern
    times = determinePatternTimes(df, pattern, windowSize)
    support = length(times)/nrow(df)
    
    # If we meet the minum support, append the selected patterns and set the
    # value to 1 at which it occurs.
    if (support>=minSupport) {
      sP[[length(sP)+1]] = pattern
      df[,paste(c("temp_pattern",pattern),collapse = "_")] = 0
      df[times,paste(c("temp_pattern",pattern),collapse = "_")] = 1
      print(pattern)
    }
  }
  return(list(df,sP))
}

determinePatternTimes = function(df, pattern, windowSize){
  times = c()
  if (length(pattern) == 1) {
    times = which(df[pattern]>0)
  } else if(length(pattern) == 3) {
    # If we have a complex pattern (<n> (b) <m> or <n> (c) <m>)
    # due to naming conventions, brackets should not be used
    # we therefore use ".b." instead of "(b)"
    timesFirstPart = determinePatternTimes(df, pattern[1], windowSize)
    timesSecondPart = determinePatternTimes(df, pattern[3], windowSize)
    if (pattern[2] == ".c.") {
      if (pattern[1]==pattern[3]) {
        # No use for co-occurences of the same patterns
        times = c()
      } else {
        times = intersect(timesFirstPart,timesSecondPart)
      } 
    } else if (pattern[2] == ".b.") {
      times = unlist(sapply(timesFirstPart, function(t){
        if(sum(t<timesSecondPart & timesSecondPart<=t+windowSize )>0) return(t)
      }))
      
    } else stop("complex operator not known")
    
  } else stop("pattern length error")
  
  return(times)
}

# extends a set of k-patterns with the 1-patterns that have sufficient support.
extendKPatterns = function(kPatterns, onePatterns) {
  newPatterns = list()
  for(kP in kPatterns) {
    for(oneP in onePatterns) {
      newPatterns[[length(newPatterns)+1]] = c(kP,".b.",oneP)
      newPatterns[[length(newPatterns)+1]] = c(kP,".c.",oneP)
    }
  }
  return(newPatterns)
}
