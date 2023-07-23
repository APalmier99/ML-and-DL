#Load the libraries required for the script
library(imager)
library(pbapply)
library(drat)
library(OpenImageR)
library(EBImage)
library(factoextra)
library(ggplot2)
library(glmnet)
library(flsa)
library(genlasso)
library(sparseSVM)
library(glinternet)
library(grpreg)
library(gglasso)
library(sparsepca)
library(elasticnet)
library(gam)
library(sparseGAM)
library(matrixcalc)
library(lars)
library(SAM)
##################################IMPORT IMAGES##################################

#Load the images on R

extract_feature <- function(dir_path, width, height) {
  
  img_size <- width*height

  #Read image
  img <- readImage(dir_path)
    
  #Resize image
  img_resized <- EBImage::resize(img, w = width, h = height)
    
  #Set to grayscale
  grayimg <- channel(img_resized, "gray")
    
  #Get the image as a matrix
  img_matrix <- grayimg@.Data
    
  #Coerce to a vector
  img_vector <- as.vector(t(img_matrix))
  return(img_vector)
}

#Images Dimension
width <- 150
height <- 150

setwd("~/Documents/Laurea Magistrale/Statistical Methods for High Dimensional Data/Project/data")

n <- length(list.files())
data <- matrix(0,nrow=n,ncol= width*height)
c <-0

for (i in list.files()){
  c <- c+1
  img <- extract_feature(dir_path = i,width = width, height = height)
  data[c,] <- img
  print(c)
}

#Response vector
y <- c(rep(1, 3725), rep(0, 3828))

#As we can see the data is balanced

##################################CHECK DATA#################################


#Let's show some of the images in the dataset

par(mfrow=c(3,3))
set.seed(4231)
examples <- sample(1:n, 9)

for (i in 1:9){
  if (y[examples[i]]==0){
    title="NO MASK"
  }
  else{
    title="MASK"
  }
  plot(Image(matrix(data[examples[i],], ncol=height, nrow=width, byrow=T)))
  title(title, font.main=2, line=5)
}

par(mfrow=c(1,1))


#Check the scale of the features (should be between 0 and 1)
sum((apply(data, 2, min)) < 0)
sum((apply(data, 2, max)) > 1)




##################################DENOISE##################################

#Function to plot the images
plot_denoise <- function(mat){
  par(mfrow=c(3,3))
  for (i in 1:3){
    plot(Image(matrix(data[reduced,][examples[i],], ncol=height, nrow=width, byrow=T)))
  }
  
  for (i in 1:3){
    plot(Image(matrix(noisy.data[examples[i],], ncol=height, nrow=width, byrow=T)))
  }
  
  
  for (i in 1:3){
    plot(Image((matrix(mat[examples[i],], ncol=height, nrow=width, byrow=T))))
  }
  par(mfrow=c(1,1))
}


#We work only on 1000 images to make everything doable

set.seed(42)
examples <- sample(1:500, 6)
reduced <- sample(1:7553 , 500)


noisy.data <- data[reduced,] + matrix(rnorm(500*width*height,mean = 0, sd = 0.2),nrow=500,ncol= width*height)
denoised.data.pca <- matrix(0,nrow=500,ncol= width*height)

for (i in 1:500){  
  print(i)
  pca.img <- prcomp(matrix(noisy.data[i,], ncol=height, nrow=width, byrow=T), center=F, scale=F)
  denoised.data.pca[i,] <- as.vector(t(pca.img$x[,1:20] %*% t(pca.img$rotation[,1:20])))
}

plot_denoise(denoised.data.pca)


#Fused Lasso

#To pick the best values of lambda we perform a grid search and check the frobenius norm
#for the first 10 images

lambda1 <- c(0,0.05, 0.1, 0.15, 0.20, 0.25)
lambda2 <- c(0,0.05, 0.1, 0.15, 0.20, 0.25)

best_i=0
best_j=0
for (n in 1:10){
  best = Inf
  for (i in lambda1){
    for (j in lambda2){
        test <- matrix(flsa(noisy.data[n,], lambda1=i, lambda2=j), ncol=width, nrow=height, byrow=T)
        frob <- frobenius.norm(data[reduced,][n,]-test)
        if (frob<=best){
          best_i =i
          best_j= j
          best=frob
        }
    }
  }
  print("Best Comb")
  print(c(n, best_i, best_j))
}

#The best combination seems to be lambda1=0 and lambda2=0.25 (for some images lambda1=0.05)
#however since this value of lambda2 makes the computation much slower we pick lambda2=0.15
#since the difference is intangible


denoised.data.fused1d <- matrix(0,nrow=500,ncol= width*height)
for (i in 1:500){
  print(i)
  denoised.data.fused1d[i,] <- flsa(noisy.data[i,], lambda1=0, lambda2=0.15)
}

plot_denoise(denoised.data.fused1d)



#Same grid search for 2d case

best_i=0
best_j=0
for (n in 1:10){
  best = Inf
  for (i in lambda1){
    for (j in lambda2){
      test <- flsa(matrix(noisy.data[n,],nrow=height,ncol=width, byrow=T), lambda1=i, lambda2=j)
      frob <- frobenius.norm(data[reduced,][n,]- matrix(test, nrow=width, ncol=height))
      if (frob<=best){
        best_i =i
        best_j= j
        best=frob
        
      }
    }
  }
  print("Best Comb")
  print(c(n, best_i, best_j, best))
}


#As earlier the best combination seems to be lambda1=0 and lambda2=0.25, however since 
#this value of lambda2 makes the computation much slower we pick lambda2=0.15 
#since the difference is intangible.

denoised.data.fused2d <- matrix(0,nrow=500,ncol= width*height)
for (i in 1:500){
  print(i)
  denoised.data.fused2d[i,] <- flsa(matrix(noisy.data[i,],nrow=height,ncol=width), lambda1=0, lambda2=0.15)
}

plot_denoise(denoised.data.fused2d)



###########################################################################





##################################ANALYSIS#################################

round(0.15*n)
indexes <- sample(1:n, round(0.15*n), replace=F)

X_train <- data[-indexes,]
y_train <- y[-indexes]

X_test <- data[indexes,]
y_test <- y[indexes]


#We try a lasso model to see how it works
lasso <- glmnet(X_train, y_train, family="binomial")
plot(lasso, xvar="lambda", xlab="Log lambda", ylab="Coefficients")

lasso.cv <- cv.glmnet(X_train, y_train, nfolds=5, family="binomial", trace.it=1)
plot(lasso.cv)
plot(lasso, xvar="lambda", xlab="Log lambda", ylab="Coefficients")
abline(v=log(lasso.cv$lambda.min), col=1)

#Look at the predictions
pred.cvlasso <- predict(lasso.cv, newx=X_test,s=lasso.cv$lambda.min, type="response")
table(round(pred.cvlasso), y_test)

#Pick indexes of those classified correctly and those classified incorrectly
set.seed(42)

pred.true <- sample(which((round(pred.cvlasso)==y_test)==T),6)
pred.false <- sample(which((round(pred.cvlasso)==y_test)==F),6)


#Let's check some of correctly classified pictures
par(mfrow=c(2,3))
for (i in 1:6){
  plot(Image(matrix(X_test[pred.true[i],], ncol=height, nrow=width, byrow=T)))
}

#Let's check some of the missclassified pictures
for (i in 1:6){
  plot(Image(matrix(X_test[pred.false[i],], ncol=height, nrow=width, byrow=T)))
}
par(mfrow=c(1,1))


#Check where are the pixels chosen
coefs <- coef(lasso, s=lasso.cv$lambda.min)[-1]!=0
coef_matrix <- matrix(coefs, nrow=height, ncol=width, byrow=T)

plot(Image(coef_matrix))


#Relaxed Lasso
train_dataframe <- data.frame(cbind(X_train[,coefs], y_train))
test_dataframe <- data.frame(cbind(X_test[,coefs], y_test))
logistic <- glm(y_train~., family="binomial", data=train_dataframe)
pred.log <- predict(logistic, newdata = test_dataframe, type="response")
table(round(pred.log), y_test)


#SparseSVM
lasso.svm <- sparseSVM(X_train,y_train)
plot(lasso.svm, xvar="lambda")

cv.svm <- cv.sparseSVM(X_train,y_train, trace=T, nfolds=5)
plot(cv.svm)
plot(lasso.svm, xvar="lambda")
abline(v=log(cv.svm$lambda.min))

pred.svm <- predict(lasso.svm, lambda=cv.svm$lambda.min, X=X_test)
table(round(pred.svm), y_test)



#Adaptive Lasso

ridge.cv <- cv.glmnet(X_train,y_train,alpha=0,family='binomial', nfolds = 5, trace.it=1)
ridge.coefs <- as.numeric(coef(ridge.cv, s = ridge.cv$lambda.min))[-1] 
w <- 1/abs(ridge.coefs)

adaptive <- glmnet(X_train,y_train,alpha=1,family='binomial', penalty.factor = w)
plot(adaptive)


cv.adaptive <- cv.glmnet(X_train,y_train,alpha=1,family='binomial', penalty.factor = w, nfolds=5, trace.it=1)
plot(cv.adaptive)

pred.adaptive <- predict(cv.adaptive, newx=X_test,s=lasso.cv$lambda.min, type="response")
table(round(pred.adaptive), y_test)


coefs.adaptive <- coef(adaptive, s=cv.adaptive$lambda.min)[-1]!=0
coef_matrix_adaptive <- matrix(coefs.adaptive, nrow=height, ncol=width, byrow=T)
par(mfrow=c(1,2))
plot(Image(coef_matrix))
plot(Image(coef_matrix_adaptive))
par(mfrow=c(1,1))
