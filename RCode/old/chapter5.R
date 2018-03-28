##########################################################################
## Hoogendoorn & Funk (2017) ML4QS Springer
## Chapter 5
## Version 17/02/01
##########################################################################

require(cluster)
require(scatterplot3d)
# Read the result from the previous chapter, and make sure the index is of the type datetime.
datasetPath = '../datasets/crowdsignals.io/csv-participant-one/'
dataset = read.csv(paste(datasetPath, 'chapter5_result.csv', sep = ""))

# Let us look at k-means first.
kValues = 2:10
silhouetteValues = c()

for (k in kValues) {
  cat("k=", k)
  fit = pam(dataset[,c('acc_phone_x', 'acc_phone_y', 'acc_phone_z')],k)
  silhouetteValues[k-1] = fit$silinfo$avg.width
}
plot(kValues,silhouetteValues,xlab="k",ylab="Silhouette Score",ylim = c(0,1),type="l",col="blue")

# And run the knn with the highest silhouette score
k = 7
tmpDat = dataset[,c('acc_phone_x', 'acc_phone_y', 'acc_phone_z')]
fit = pam(tmpDat,k)
pdf("chapter5plots.pdf")
plot(fit)
dev.off()

tmp =dataset[,colnames(dataset)[grepl("label",substr(colnames(dataset),1,10))]]*matrix(rep(1:8,2895),ncol=8,byrow = TRUE)
tmpDat$pch = apply(tmp,1,max)

with(tmpDat, {
  scatterplot3d(acc_phone_x, acc_phone_y, acc_phone_z,
                color= rainbow(k)[fit$clustering], # color refers to cluster
                pch=pch, # marker refers to task label
                main="")
})

# Hierarchical clustering 
cluster = hclust(dist(tmpDat[,-4]),method = "average")
kMax = 20
memb = cutree(cluster, k = kMax)
cent = NULL
for(k in 1:kMax){
  cent <- rbind(cent, colMeans(tmpDat[,-4][memb == k, , drop = FALSE]))
}
hc1 <- hclust(dist(cent)^2, method = "average", members = table(memb))
plot(hc1)
