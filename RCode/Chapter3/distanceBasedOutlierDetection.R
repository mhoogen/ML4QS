##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   ./chapter3/distanceBasedOutlierDetection.R
# 
##########################################################################

# The most simple distance based algorithm. We assume a distance function, e.g. 'euclidean'
# and a minimum distance of neighboring points and frequency of occurrence.
simpleDistanceBased = function(df, cols, dmin, fmin, dFunction = "euclidean") {
  # Normalize data
  data = scale(df[,cols])
  data = data[!is.na(data)]

  d = as.matrix(dist(data, method = dFunction))
  df[!is.na(df[,col]),paste(cols,"_simpleDistOutlier",sep="")] = sapply(1:nrow(d), function(x) {(sum(d[x,-x]>dmin)/nrow(d))>fmin})
  return(df)
}

# Compute the local outlier factor. K is the number of neighboring points considered, d_function
# the distance function again (e.g. 'euclidean').
localOutlierFactor = function(df, cols, dFunction = "euclidean", k, plot = FALSE) {
  data = scale(df[,cols])
  data = data[!is.na(data)]
  
  df[!is.na(df[,col]),paste(cols,"_lofOutlier",sep="")] = as.data.frame(lof(data,k,method=dFunction))
  if(plot)  { 
    plot.ecdf(df$lof, xlim = c(0,10),main = "LOF diagnostic")
    grid()
  }
  return(df)
}