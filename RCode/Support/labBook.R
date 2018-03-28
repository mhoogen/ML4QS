##########################################################################
## Hoogendoorn & Funk (2017) ML4QS Springer
## Lab Book
## Version 17/02/10
##########################################################################

source("./Support/util.R")

## quick aggregation of time series
require(data.table)
require(bit64)
require(zoo)
datasetPath = '../datasets/crowdsignals.io/csv-participant-one/'
filename = 'accelerometer_phone.csv'
#filename = 'gyroscope_phone.csv'
#filename = 'heart_rate_smartwatch.csv'

ds = loadDataset(datasetPath,filename)
tempTS = ds$timestamps[1]
ds[,timestamps:=as.POSIXct(ds[,timestamps], origin="1970-01-01")]
ts1 = zoo(ds[,c("x","y","z"),with=FALSE],order.by = ds[,timestamps])

aggWindow = 0.05 # aggregates on X sec intervals
ts2 = aggregate(ts1, time(ts1) - as.numeric(time(ts1)) %% aggWindow, mean)  
time(ts2[1:100,])-time(ts2[2:101,])
plot(ts2[40000:42000,"x"],xlab="time",ylab="acc_x")
plot(ts2[40000:40200,"x"],xlab="time",ylab="acc_x")

#ts3 = ts(stats::filter(ts2$x,sides = 2, rep(1/10,10)),frequency=1/aggWindow)
#ts3 = zoo(ts3,order.by = as.POSIXct(tempTS+(0:(length(ts3)-1))*aggWindow,origin="1970-01-01"))

# https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/
limTS = c(41000:42000)
acf((ts2[limTS,"x"]),lag.max = 100)
pacf((ts2[limTS,"x"]),lag.max = 100)
(fit <- arima((ts2[limTS,"x"]),order = c(0,1,1), seasonal = list(order = c(0, 1, 0),period=22)))
pred= predict(fit,n.ahead = 100)
plot(1:100,as.numeric(ts2[41901:42000,"x"]), xlim=c(0,200), ylim=c(-15,15), ylab="x", xlab="time",type="l")
lines(101:200,pred$pred,lty=3)
lines(101:200,pred$pred+pred$se,lty=1,col="grey")
lines(101:200,pred$pred-pred$se,lty=1,col="grey")
grid()


# Basic ts simulation and fitting
temp.ts= arima.sim(n = 1000, list(order = c(1,0,1), ar = c(-.9),ma=0.3))
acf(temp.ts,lag.max = 500)
pacf(temp.ts,lag.max = 50)
fit=arima(temp.ts,order = c(1,0,1))
plot(forecast(fit,h = 10),include = 20)


# Applying to crowdsignal raw data
temp.ts = ds$x[400000:402000]
acf(temp.ts,lag.max = 1000)
pacf(temp.ts,lag.max = 200)
# sarima.for(temp.ts,600,1,0,0,1,0,0,221)

fit = auto.arima(temp.ts,max.p = 10,max.q = 10,seasonal = TRUE,stepwise = FALSE)
summary(fit)

plot.ts(temp.ts)
lines(0:2000,fit$fitted,col="red")

plot(forecast(fit,h = 40),include =400)
lines(2001:2400,ds$x[402001:402400],col="red")
lines(2001:2400,ds$x[401779:402178],col="green",lwd=2)

# Applying to crowdsignal aggregated data
load("50msAggregatedAccData.RData")
temp.ts = dataset1$acc_phone_x[40001:40200]
acf(temp.ts,lag.max = 100)
pacf(temp.ts,lag.max = 50)

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
sarima.for(temp.ts,40,3,0,3,1,0,0,22)

fit = arima(temp.ts, order = c(3,0,3), seasonal = list(order=c(1,0,0),period = 22), include.mean = FALSE)
summary(fit)
plot(forecast(fit,h = 40),include=50,main="",xlab = "t",ylab = "x-acceleration")
lines(201:240,dataset1$acc_phone_x[40201:40240],col="red")
grid()
