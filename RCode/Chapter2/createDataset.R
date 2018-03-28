##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   ./Chapter2/createDataset.R
# 
##########################################################################

addNumericalDataset = function(df, filename, timestampCol, valueCols, func, prefix, path, timeWindow) {
# can be implemented in a more efficient manner using the aggregate() function on the data.table itself,
# however, for the sake of comparability it is implemented as in the Python code
  print(paste("Load: ",path,filename,sep=""))
  dataset = fread(paste (path,filename,sep=""))
  tics2millisec = 1000000
  dataset[,(timestampCol):=dataset[,timestampCol,with=FALSE]/tics2millisec]
  
  cNames = paste(prefix,valueCols,sep="")

  if (is.null(df)) {
    # create aggregated data.frame and time intervals
    tMin = min(dataset[,timestampCol,with=FALSE])
    tMax = max(dataset[,timestampCol,with=FALSE])
    timeInts = tMin + timeWindow*(0:(floor(tMax-tMin)/timeWindow))
    
    df = data.frame(time = timeInts)
  }
  df[,cNames] = NA
  
  j = 1
  for (t in df$time) {
    relevantRows = dataset[timestamps>=t&timestamps<t+timeWindow,valueCols,with = FALSE]
    df[j,cNames]=apply(relevantRows,2,mean)[valueCols]
    j = j + 1
  }
  return(df)
}

addEventDataset = function(df, filename, startTimeCol, endTimeCol, valueCol, aggregate = "sum", path) {
  if (is.null(df)) {
    stop("Provide a non-NULL data.frame (add numerical data first)")
  }
  print(paste("Load: ",path,filename,sep=""))
  dataset = fread(paste (path,filename,sep=""))
  tics2millisec = 1000000
  dataset[,(startTimeCol):=dataset[,startTimeCol,with=FALSE]/tics2millisec]
  dataset[,(endTimeCol):=dataset[,endTimeCol,with=FALSE]/tics2millisec]
  dataset[,(valueCol):=sapply(dataset[,valueCol,with=FALSE],FUN = function(x) {paste(valueCol,gsub("[^[:alnum:]]", "", x),sep="")})]
  eventValues = unlist(unique(dataset[,valueCol,with=FALSE]))
  df[,eventValues] = 0
  
  start = sapply(dataset[,startTimeCol,with = FALSE],as.numeric)
  end = sapply(dataset[,endTimeCol,with = FALSE],as.numeric)
  value = sapply(dataset[,valueCol,with=FALSE],as.character)
  for (i in 1:nrow(start)) {
    df[start[i]<=df$time & end[i]>df$time,value[i]] = 1 #which.max(value[i] == eventValues)
  }
  return(df)
}