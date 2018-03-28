##########################################################################
## Hoogendoorn & Funk (2017) ML4QS Springer
## Chapter 6 - Gradient Field
## Version 17/09/04
##########################################################################

require(ggplot2)

minX <- 0
maxX <- 5
minY <- 0
maxY <- 5
len <- 30

data <- data.frame(expand.grid(x = seq(minX, maxX, length.out=len), 
                               y = seq(minY, maxY, length.out=len)))  
data$dx = data$dy = rep(0,ncol(data))

for (i in 1:nrow(data)) {
  data$dx[i] = t(X%*%c(1,data$x[i],data$y[i])-Y)%*%X[,2]/N
  data$dy[i] = t(X%*%c(1,data$x[i],data$y[i])-Y)%*%X[,3]/N
}

p = ggplot(data=data, aes(x=x,y=y), environment = environment()) + 
  geom_point(size = 0.5) + 
  geom_segment(aes(xend=x+dx, yend=y+dy), arrow = arrow(length = unit(0.1, "cm"))) + 
  xlim(minX,maxX) + 
  ylim(minY,maxY) +
  xlab(expression(theta[1])) +
  ylab(expression(theta[1]))

print(p)