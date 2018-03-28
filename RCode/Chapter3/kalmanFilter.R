##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   ./chapter3/kalmanFilter.R
# 
##########################################################################

# There are a number of R packages which implement Kalman filtering
# see: Tusell, Fernando. "Kalman filtering in R." Journal of Statistical Software 39.2 (2011): 1-27
# in addition to that there are "build-in" functions as part of the stats package that implement Kalman filtering
# e.g. KalmanSmooth etc.
# here we use the KFAS package, which is a recent and extensive implementation of Kalman filtering for forecasting and smoothing

applyKalmanFilter = function(df, col) {
  dat = ts(df[,col])
  
  model = SSModel(dat~SSMtrend(1,Q=list(matrix(NA))),H=matrix(NA))
  model = fitSSM(inits=c(log(var(dat,na.rm = TRUE)),log(var(dat,na.rm = TRUE))),model=model,
                    method='BFGS',control=list(REPORT=1,trace=1))$model
  out = KFS(model,filtering='state',smoothing='state')$a
  
  df[,paste(col,"_kalman",sep="")] = out[1:(length(out)-1)]
  return(df)
}