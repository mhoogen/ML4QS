##########################################################################
## Hoogendoorn & Funk (2017) ML4QS Springer
## Chapter 6 - Generate data for linear regression
## Version 17/02/01
##########################################################################

generateRegressionData =
  function(thetaTrue,N,error=1,xMean=0) {
    # Purpose
    #   generate data for linear regression
    # Arguments
    #   thetaTrue: vector of true values for the coefficients of the linear function, 
    #     thetaTrue[1] is the intercept (=\theta_0)
    #   N: number of samples to be generated
    #   error: error added to linear output
    #   xMean: represents the shift of the design matrix
    # Output
    #   X: design Matrix N*length(thetaTrue)
    #   Y: vector
    
    if(missing(thetaTrue)) stop("thetaTrue is missing")
    if(missing(N)) stop("N = number of samples to be generated is missing")
    
    # generate design matrix X
    X = cbind(rep(1,N),matrix(runif(N*(length(thetaTrue)-1),xMean-0.5,xMean+0.5),nrow = N))
    # calculate output y
    Y = X %*% thetaTrue + rnorm(N, sd = error)
    return(list(X=X,Y=Y))
  }

