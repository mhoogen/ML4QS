### Gradient descent
## Chapter 2 - Section \ref{chap2:sec:learning}
## 16/09/04

widrowHoff = 
  function(Y,X,error=1e-7,steps=10000,eta=2e-4) {
    # Purpose
    #   implement batch gradient descent
    # Arguments
    #   X: design Matrix N*length(thetaTrue)
    #   Y: vector
    # Output
    #   thetaHat: thetaHat minimizes the in-sample error
    #   thetaHat_interate: contains the sequence of thetaHat
    
    thetaHat = rep(0,ncol(X)) # arbitrary starting point
    thetaHat_iterate = thetaHat
    s = 1 
    XtX = t(X)%*%X
    XtY = t(X)%*%Y
    repeat {
      delta  = eta*(XtX%*%thetaHat-XtY)/ncol(X)
  
      thetaHat = thetaHat - delta
      thetaHat_iterate <- rbind(thetaHat_iterate,t(thetaHat))
      s = s +  1
      if (t(delta)%*%delta < error || s>steps) break
    }
  return(list(thetaHat=thetaHat,iterations=s,delta=delta,thetaHat_iterate=thetaHat_iterate)) 
}

set.seed(1234)
N = 1000
thetaTrue = c(1,3,2)

# To make the plot a bit more interesting X is not demeaned (xMean = 0.5)
regData = generateRegressionData(thetaTrue,N,xMean=0.5)
X = regData$X
Y = regData$Y

res = widrowHoff(Y,X)
nGrid = 100
nSize = nGrid/5
g = expand.grid(theta1=(0:nGrid)/nSize,theta2=(0:nGrid)/nSize)
z = rep(NA,nrow(g))
for (i in 1:nrow(g)) {
  z[i] = t(Y-X%*%c(1,g[i,1],g[i,2]))%*%(Y-X%*%c(1,g[i,1],g[i,2]))/N
}

g$z = z  
par(mfrow=c(1,1))
contour((0:nGrid)/nSize,(0:nGrid)/nSize,matrix(g$z,nGrid+1,nGrid+1),nlevels=15,xlab=expression(theta[1]),ylab=expression(theta[2]))
lines(res$thetaHat_iterate[,2:3],col=3,lwd=3)
points(res$thetaHat_iterate[,2:3], pch=19,cex=0.5)
grid()

# filled.contour((0:nGrid)/nSize,(0:nGrid)/nSize,matrix(g$z,nGrid+1,nGrid+1),xlab=expression(theta[1]),ylab=expression(theta[2]))
