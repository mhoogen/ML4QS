# TS plot

require(data.table)
require(bit64)
require(zoo)

rm(list=ls())
#setwd("../datasets/crowdsignals.io/Csv-merged/")
accelerometer = fread("merged_accelerometer_1466120659000000000_1466125720000000000.csv")

tics2sec = 1000000000
timeAtts = c("start","end","timestamps")
accelerometer=accelerometer[,c(timeAtts,"x","y","z"),with=FALSE]
accelerometer[,timeAtts:=accelerometer[,timeAtts,with=FALSE] / tics2sec,with=FALSE]

# THIS PART DOES THE TRICK
# this converts the data.table into a time series, aggregates and plots it
accelerometer[,timestamps:=as.POSIXct(accelerometer[,timestamps], origin="1970-01-01")]
ts1 = zoo(accelerometer[,c("x","y","z"),with=FALSE],order.by = accelerometer[,timestamps])
plot(aggregate(ts1, time(ts1) - as.numeric(time(ts1)) %% 60, mean),main="",xlab="time")
