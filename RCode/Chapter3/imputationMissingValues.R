##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   ./chapter3/imputationMissingValues.R
# 
##########################################################################

# Imputation is covered in different R packages, one is mice - it offers lots of methods
# see methods(mice)

imputeMean = function(df,col) {
  df[is.na(df[,col]),col] = mean(df[,col],na.rm = TRUE)
  return(df)
}

imputeMedian = function(df,col) {
  df[is.na(df[,col]),col] = median(df[,col],na.rm = TRUE)
  return(df)
}

# for the following function there are other existing functions in the zoo package like na.spline. 
# Other packages such as imputeTS also supports imputation for time series
imputeInterpolation = function(df,col) {
  dat = zoo(df[,col])
  dat = ifelse(is.nan(dat),NA,dat)
  # fill initial and trailing NA
  if (is.na(dat[1])) dat[1] = dat[which.max(!is.na(dat))]
  if (is.na(dat[length(dat)])) dat[length(dat)] = dat[max((!is.na(dat))*(1:length(dat)))]
  # interpolate
  df[,col] = na.approx(dat)
  return(df)
}