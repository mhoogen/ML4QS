##########################################################################
## Hoogendoorn & Funk (2017) ML4QS Springer
## Chapter 6 - In-sample error 
## Version 16/09/04
##########################################################################

library("lattice")

set.seed(1234)
N = 1000
thetaTrue = c(1,3,2)

# generate data, X will have expected mean of 0, if xMean = 0 (default value)
regData = generateRegressionData(thetaTrue,N)
X = regData$X
Y = regData$Y

g = expand.grid(theta1=(0:25)/5,theta2=(0:25)/5)
z = rep(NA,nrow(g))
for (i in 1:nrow(g)) {
  # to generate the plot we fix theta_0 = 1
  z[i]<-t(Y-X%*%c(1,g[i,1],g[i,2]))%*%(Y-X%*%c(1,g[i,1],g[i,2]))/N
}

trellis.par.set("axis.line",list(col=NA,lty=1,lwd=1))
p = wireframe(z ~ theta1 * theta2, data = g, 
          scales = list(arrows = FALSE),
          drape = TRUE, colorkey = TRUE,
          screen = list(z = 30, x = -60),
          col.regions = colorRampPalette(c("white", "green","red","red"))(100),
          xlab=expression(theta[1]),ylab=expression(theta[2]),zlab=expression(E["in"](h)))
print(p)
