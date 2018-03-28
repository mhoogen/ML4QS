##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   ./chapter3/distributionBasedOutlierDetection.R
# 
##########################################################################

# Finds outliers in the specified column of datatable and adds a binary columns with
# the same name extended with '_outlier' that expresses the result per data point.
require(mclust) 
chauvenet = function(df, col) {
  data = df[,col]
  data = data[!is.na(data)]
  mean = mean(data)
  std = sd(data)
  n = nrow(df)
  criterion = 1.0/(2*n)
  deviation = abs(data - mean)/std
  
  df[!is.na(df[,col]),paste(col,"_chauvenetOutlier",sep="")] = ((1-pnorm(deviation))<criterion)
  return(df)
}

# Fits a mixture model towards the data expressed in col and adds a column with the probability
# of observing the value given the mixture model.
mixtureModel = function(df,col, numOfComponents = NULL, plot = FALSE) {
  data = df[,col]
  data = data[!is.na(data)]
  mclBIC = mclustBIC(data)
  mod = densityMclust(data, G = numOfComponents)
  
  if (plot) {
    # some diagnostic plots
    plot(mclBIC) # determine optimal number of clusters for equal (E) and variable (V) variance
    grid()
    plot(mod, what = "density", data = data, xlab = col)
    grid()
    plot(mod, what = "diagnostic", type = "cdf")
  }
  
  df[!is.na(df[,col]),paste(col,"_mixtureProb",sep="")] = sapply(data,function(x) {
    sum(mod$parameters$pro*dnorm(rep(x,mod$G),
                                 mod$parameters$mean,
                                 sqrt(mod$parameters$variance$sigmasq)))
    })
  return(df)
}
