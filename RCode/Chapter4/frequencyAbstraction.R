##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   Chapter 4 - Feature Engineering based on Sensory Data
#   ./chapter4/frequencyAbstraction.R
# 
##########################################################################

abstractFrequency = function(df, cols, windowSize, samplingRate, instances2Analyze = NULL) {
  
  # create new columns for the frequency data
  if(is.null(instances2Analyze)) instances2Analyze = 1:nrow(df)
  freqs = (0:(windowSize-1))*samplingRate/windowSize

  for (col in cols) {
    df[,paste(col,"_max_freq",sep="")] = NA
    df[,paste(col,"_freq_weighted",sep="")] = NA
    df[,paste(col,"_pse",sep="")] = NA
    for (freq in freqs) {
      df[,paste(col,"_freq_",freq,"_Hz_ws_",windowSize, sep="")] = NA
    }
  }
  
  # pass over the dataset (we cannot compute it when we do not have enough history)
  # and compute the values
  instances2Analyze = instances2Analyze[instances2Analyze>windowSize]
  k = 1
  for (i in instances2Analyze) {
    if(k%%100 ==0) print(paste("Instance: ",i))
    k = k +1
    for (col in cols) {
      dft = fft(df[(i-windowSize):(i-1),col])
      for (j in 1:length(freqs)) {
        df[i,paste(col,"_freq_",freqs[j],"_Hz_ws_",windowSize, sep="")] = Re(dft[j])
      }
      df[i,paste(col,"_max_freq",sep="")] = freqs[which.max(Re(dft))]
      df[i,paste(col,"_freq_weighted",sep="")] =  sum(freqs * Re(dft)) / sum(Re(dft))
      pse = Re(dft)^2/windowSize
      pse = pse/sum(pse)
      df[i,paste(col,"_pse",sep="")] = -sum(log(pse)*pse)
    }
  }
  return(df)
}

# Remove periodic sinusoid functions from our data to be left with the "clean" signal
# The components should be specified by means of their index (meaning their period).
# removeComponents = function(df, col, components = 0) {
#   dft = fft(df[,col])
#   dft[components] = 0
#   df[,col] = Re(fft(dft, inverse = TRUE)/length(dft))
#   return(df)
# }
