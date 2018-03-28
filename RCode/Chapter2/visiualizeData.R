##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   ./chapter2/visualizeData.R
# 
##########################################################################

# Plot the dataset, here columns can specify a specific attribute, but also a generic name that occurs
# among multiple attributes (e.g. label which occurs as labelWalking, etc). In such a case they are plotted
# in the same graph.

require(ggplot2)

plotDataset = function(df, columns, match) {
  cNames = colnames(df)
  p = list()
  if (is.null(match)) match = rep("like",length(columns))
  for (i in 1:length(columns)) {
    if(match[i] == "exact") {
      selCols = columns[i]
    } else {
      selCols = cNames[grepl(columns[i],cNames)]
    }
    dfTemp = df[,c("time",selCols)]
    dfTemp = melt(dfTemp,id.vars = c("time"))
    p[[i]] = plot_ly(dfTemp,x = ~time, y = ~value) %>%
      add_lines(color = ~variable )
  }
  return(p)
}

# Plot outliers in case of a binary outlier score. Here, the col species the real data column and outlier_col
# the columns with a binary value (outlier or not)
plotBinaryOutliers = function(df, col, outlier_col) {
  plot(df$time,df[,col],type="l", col = "blue" ,ylab = "value", xlab = "time",main = outlier_col)
  points(df[df[,outlier_col],"time"] ,df[df[,outlier_col],col],pch = 3, col = "red",ylab = "value", xlab = "time")
  legend("topright",c(paste("outlier",col),paste("no outlier",col)), pch = 3, col = c("red", "blue"), cex = 1)
  grid()
}

plotImputedValues = function(df,col,imputedList) {
  opar = par()
  par(mfrow= c(length(imputedList),1))
  for (i in 1:length(imputedList)) {
    plot(df$time,df[,col],pch = 20,col = "blue",ylab=paste(""),xlab="time")
    points(df$time[is.na(df[,col])],imputedList[[i]][is.na(df[,col]),col],pch=20,col="red")
    print(colnames(imputedList[[i]]))
  }
  par(opar)
}

plotBoxplot = function(df, col) {
  ggplot(stack(df[,col]), aes(x = ind, y = values)) +
    geom_boxplot()  +
    xlab("") + ylab("")
}
