##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   ./chapter3/principalComponent.R
# 
##########################################################################


pcaExplainedVariance = function(df, cols) {
  pca = prcomp(df[,cols],scale = TRUE)
  prVariance = pca$sdev^2
  return(prVariance/sum(prVariance))
}

applyPCA = function(df, cols, nComponents) {
  pca = prcomp(df[,cols], retx = TRUE, center = TRUE, scale = TRUE)
  for (comp in 1:nComponents) {
    df[,paste("pca_", as.character(comp), sep="")] = pca$x[,comp]
  }
  return(df)
}