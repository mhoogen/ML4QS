##########################################################################
#
#   Mark Hoogendoorn & Burkhardt Funk (2017)
#   Machine Learning for the Quantified Self
#   Springer
#   Chapter 5 - Clustering
#   ./crowdsignals_ch5.R
# 
##########################################################################

rm(list=ls())
require(cluster)
require(scatterplot3d)
require(ggplot2)

### settings and data import
resultPath = "./intermediate_datafiles/"
load(file=paste(resultPath,"chapter4_result.RData",sep=""))

# compute the number of milliseconds covered by an instane based on the first two rows
timeWindow = as.numeric(difftime(crowdSignalData$time[2],crowdSignalData$time[1],units = "secs"))*1000

### k-means clustering

# check silhouette scores for k-means with different k
kValues = 2:9
silhouetteValues = c()

tmpDat = crowdSignalData[,c('acc_phone_x', 'acc_phone_y', 'acc_phone_z')]
for (k in kValues) {
  cat("k=", k,"\n")
  resKM = kmeans(tmpDat, centers = k, 
                 iter.max = 50, nstart = 50)
  dai = daisy(tmpDat)
  sil = silhouette(resKM$cluster,dai) 
  silhouetteValues[k-kValues[1]+1] = summary(sil)$avg.width
}
plot(kValues,silhouetteValues,xlab="k",ylab="Silhouette Score  (K-Means)",ylim = c(0,1),type="l",col="blue")
grid()

# run k-means with the highest silhouette score
k = 6
resKM = kmeans(tmpDat, centers = k, 
               iter.max = 50, nstart = 50)
sil = silhouette(resKM$cluster,dai) 
factoextra::fviz_silhouette(sil)

# plot labels and clusters in 3D
tmp =crowdSignalData[,colnames(crowdSignalData)[grepl("label",substr(colnames(crowdSignalData),1,10))]]
tmpDat$pch = apply(tmp,1,max,na.rm=TRUE)
with(tmpDat, {
  scatterplot3d(acc_phone_x, acc_phone_y, acc_phone_z,
                color= rainbow(k)[resKM$cluster], # color refers to cluster
                pch=pch, # marker refers to task label
                main="")
})

# clustering using k-medoids
tmpDat = crowdSignalData[,c('acc_phone_x', 'acc_phone_y', 'acc_phone_z')]
kValues = 2:9
silhouetteValues = c()

for (k in kValues) {
  cat("k=", k,"\n")
  resKMed = pam(tmpDat, k)
  dai = daisy(tmpDat)
  sil = silhouette(resKMed$cluster,dai) 
  silhouetteValues[k-kValues[1]+1] = summary(sil)$avg.width
}
plot(kValues,silhouetteValues,xlab="k",ylab="Silhouette Score (K-Medoids)",ylim = c(0,1),type="l",col="blue")
grid()

# run k-medoids with the highest silhouette score
k = 6
resKMed = pam(tmpDat, k)
sil = silhouette(resKMed$cluster,dai) 
factoextra::fviz_silhouette(sil)

# demonstrate that the k-means and k-medoids lead to very similar clusters
table(resKM$cluster,resKMed$clustering) # cluster numbers are different, but otherwise no big difference

### hierarchical clustering (agglomerative) 
cluster = hclust(dist(tmpDat[,-4]),method = "ward.D")
kMax = 6
memb = cutree(cluster, k = kMax)
sil = silhouette(memb,dai) 
factoextra::fviz_silhouette(sil)

cent = NULL
for(k in 1:kMax){
  cent <- rbind(cent, colMeans(tmpDat[,-4][memb == k, , drop = FALSE]))
}
hc1 <- hclust(dist(cent)^2, method = "average", members = table(memb))
plot(hc1)

# write output
crowdSignalData$cluster = resKM$cluster
resultPath = "./intermediate_datafiles/"
save(crowdSignalData, file=paste(resultPath,"chapter5_result.RData",sep=""))
