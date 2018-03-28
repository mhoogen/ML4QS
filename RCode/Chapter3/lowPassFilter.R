##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   ./chapter3/lowPassFilter.R
# 
##########################################################################

require(signal)

lowPassFilter = function(df, col, samplingFrequency, cutoff, nOrder) {
  nyquistFrequency = 0.5 * samplingFrequency # Nyquist frequency := half of the sampling frequency
  W = cutoff/nyquistFrequency 
  butterworthFilter = butter(nOrder,W)
  df[,paste(col,"_lowpass",sep="")] = filtfilt(butterworthFilter,df[,col])
  return(df)
}