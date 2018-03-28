##########################################################################
## Hoogendoorn & Funk (2017) ML4QS Springer
## Chapter 8
## Version 17/02/01
##########################################################################

## Figure 8.2 - three types of autocorrelation
set.seed(42)
randomTS = rnorm(5200,mean = 0.1, sd = 2)
par(mfrow=c(3,2))
plot.ts(randomTS[101:500],main="",xlab = "Time" , ylab = "Measures")
acf(randomTS[101:500],main="")

z = stats::filter(randomTS,sides=2,rep(1/10,10))
plot.ts(z[101:500],main="",xlab = "Time" , ylab = "Measures")
acf(z[101:500],main="")

z = cumsum(randomTS)
plot.ts(z[101:5100],main="",xlab = "Time" , ylab = "Measures")
acf(z[101:5100],main="")

## Figure 8.3 - Exponential Smoothing
par(mfrow=c(1,1))
plot.ts(z[1:1000],ylab = "Measures")
alpha = 0.2
lines(stats::filter(z[1:1000],sides = 2,(1-alpha)^c(100:0,1:100)/(2-alpha)*alpha),ylab = "Measures",col="blue",lty = 3, lwd =2)
alpha = 0.05
lines(stats::filter(z[1:1000],sides = 2,(1-alpha)^c(100:0,1:100)/(2-alpha)*alpha),ylab = "Measures", col="red",lty = 2,lwd=2)
grid()

## Figure 8.4 - Differencing
z = rnorm(5200) + c(rep(0,1000),rep(20,1000),20-c(1:2000)/100,rep(0,1200))+5
plot.ts(z,ylab="Measures",ylim=c(-5,30))
grid()
alpha = 0.05
lines(stats::filter(z,sides = 2,(1-alpha)^c(100:0,1:100)/(2-alpha)*alpha), col="red",lty = 2,lwd=2)
lines(stats::filter(z,sides = 2,c(rep(0,100),1,rep(0,100))-(1-alpha)^c(100:0,1:100)/(2-alpha)*alpha),col="blue",lty = 3,lwd=1)

## Figure 8.15
opar = par()
par(mar=c(5,4,4,0.5),mfrow=c(1,2))
plot(dataset1$time,stats::filter(dataset1$acc_phone_x,sides = 2,c(1)),ylim=c(-25,35),type="l",ylab="Acceleration",xlab="Time")
grid()
plot(dataset1$time,stats::filter(dataset1$acc_phone_x,sides = 2,c(1,-1)),ylim=c(-25,35),type="l",ylab="",xlab="Time")
grid()
par(opar)

## Figure 8.16
load("50msAggregatedAccData.RData")

# cartesian coords
xfil = stats::filter(dataset1$acc_phone_x,sides = 2,rep(1/201,201))
yfil = stats::filter(dataset1$acc_phone_y,sides = 2,rep(1/201,201))
zfil = stats::filter(dataset1$acc_phone_z,sides = 2,rep(1/201,201))

# polar coords
radius = sqrt(xfil^2+yfil^2+zfil^2)
theta = acos(zfil/radius)

phi = rep(NA,length(xfil))
sub = xfil>0 & !is.na(xfil)
phi[sub] = atan(yfil[sub]/xfil[sub])
sub = xfil==0 & !is.na(xfil)
phi[sub] = sign(yfil[sub])*pi/2 
sub = xfil<0 & yfil>=0 & !is.na(xfil)
phi[sub] = atan(yfil[sub]/xfil[sub])+pi
sub = xfil<0 & yfil<0 & !is.na(xfil)
phi[sub] = atan(yfil[sub]/xfil[sub])-pi

plot(dataset1$time,dataset1$acc_phone_x,type="l",ylab="Acceleration",xlab="Time",col="grey")
lines(dataset1$time,xfil,col="red")
lines(dataset1$time,radius,col = "blue")
grid()
legend("bottomright",c("x(plain)","x(filtered)",paste(expression("||a||"),"(filtered)")), lty = c(1,1,1),
       col=c('grey','red','blue'))

## 3d coordinates
require(scatterplot3d)
st = 15000
en = 20000
coord = data.frame(x = xfil[st:en],y = yfil[st:en],z = zfil[st:en])
scatterplot3d(coord$x,coord$y,coord$z, main="",pch=20,highlight.3d = TRUE)

## Fig. 8.17
source("./Support/util.R")
require(data.table)
require(bit64)
require(zoo)
require(forecast)
datasetPath = '../datasets/crowdsignals.io/csv-participant-one/'
filename = 'accelerometer_phone.csv'
ds = loadDataset(datasetPath,filename)
ds$timestamps = as.POSIXct(ds$timestamps, origin="1970-01-01")
temp.ts = ds$x[400000:402000]

opar = par()
par(mar=c(5,4,4,0.5),mfrow=c(1,2))
acf(temp.ts,lag.max = 1000,main="",ylab="autocorrelation function (ACF)",xlab="lag")
grid()
pacf(temp.ts,lag.max = 100,main="",ylab="partial autocorrelation function (PACF)",xlab="lag")
grid()
par(opar)
# sarima.for(temp.ts,600,1,0,0,1,0,0,221)

## Fig. 8.18
fit = auto.arima(temp.ts,max.p = 10,max.q = 10,seasonal = TRUE,stepwise = FALSE)
summary(fit)
plot.ts(temp.ts[1:500],main="",ylab="acc_x")
grid()
lines(1:500,fit$fitted[1:500],col="red")
lines(temp.ts[1:500]-fit$fitted[1:500],col="blue")
legend("topleft",c("original","prediction","difference"), lty = c(1,1,1),
       col=c('black','red','blue'))

## Fig. 8.19
plot(forecast(fit,h = 40),include =100,main="",ylab ="acc_x",xlab="Time")
lines(2001:2400,ds$x[402001:402400],col="red")
#lines(2001:2400,ds$x[401779:402178],col="green",lwd=2)
grid()
  

# Fig. 8. 20
load("50msAggregatedAccData.RData")
temp.ts = dataset1$acc_phone_x[40001:40200]

aicMin =1e20
for(p in 0:3) {
  for(q in 0:3) {
    fit = arima(temp.ts,order = c(p,0,q),seasonal = list(order = c(1,0,0),period=22))
    if(fit$aic <aicMin) {
      aicMin=fit$aic
      minFit = fit
      print(paste(p,q,aicMin))
    }
  }
}
# nice function from http://lib.stat.cmu.edu/general/tsa2/Rcode/itall.R
# sarima.for(temp.ts,40,3,0,3,1,0,0,22)

summary(minFit)
plot(forecast(minFit,h = 40),include=50,main="",xlab = "Time",ylab = "acc_x")
lines(201:240,dataset1$acc_phone_x[40201:40240],col="red")
grid()
legend("topleft",c("original","prediction","true values"), lty = c(1,1,1),
       col=c('black','blue','red'))