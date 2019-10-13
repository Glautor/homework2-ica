library(e1071)
library(ggplot2)
library(corrplot)
library(factoextra)
library(Metrics)
library(glmnet)
library(tidyverse)
library(caret)
library(foba)
library(ridge)

setwd("C://your-directory")
#                                               Parte 0 - Pre Processamento
# o treino_y e teste_y Ã© a solubilidade de cada composto
treino_x = read.table("solTrainX.txt")
treino_y = read.table("solTrainY.txt")
teste_x = read.table("solTestX.txt")
teste_y = read.table("solTestY.txt")
dim(treino_x)
dim(treino_y)
dim(teste_x)
dim(teste_y)
dataset = rbind(treino_x,teste_x)
dataset["y"] = rbind(treino_y,teste_y)
# sÃ£o 208 Fingerprints(indicadores de molÃ©culas), entÃ£o as Features de interesse sÃ£o as 20 restantes
Cor = cor(dataset[209:228])
dataTreino = treino_x
dataTreino["y"] = treino_y
dataTeste = teste_x
dataTeste["y"] = teste_y
# a matriz de correlaÃ§Ã£o, onde hÃ¡ 34 colunas com correlaÃ§Ã£o > 0.9
png("Matriz_CorrelaÃ§Ã£o_circulos.png")
corrplot(Cor, method = "number")
dev.off()

RemoverCorr = findCorrelation(cor(dataTreino), .9) # ==> 34 elementos
dataTreino = dataTreino[, -RemoverCorr]
dataTeste = dataTeste[, -RemoverCorr]
# Colunas finais removidas : NumAtoms, NumNonHAtoms, NumBonds, NumNonHBonds, NumMultBonds, NumHalogen, SurfaceArea2



#PCA
pca.data = prcomp(treino_x[1:228],scale=TRUE)
pdf("VariÃ¢ncia vs Componentes.pdf")
fviz_eig(pca.data,xlab="Componente",ylab="VariÃ¢ncia",addlabels = TRUE,main="VariÃ¢ncia vs Componentes")
dev.off()

ggplot(dataset,aes(dataset[,1])) + geom_histogram(color="black",fill="springgreen2")+theme_gray()+
    labs(x=names(dataset[1]),y="FrequÃªncia")
# Scatter plot
setwd("C://your-directory")
for(i in 1:228){
  
  ggplot(dataset,aes(dataset[,i])) + geom_histogram(color="black",fill="springgreen2")+theme_gray()+
    labs(x=names(dataset[i]),y="FrequÃªncia")
  ggsave(paste(names(dataset[i]),".pdf"))
}
  dev.off()
  setwd("C://your-directory")


#                                               Parte 1 - Ordinary Linear Regression

#                                               Parte 2 - LÂ² Penalized Linear Regression

# Build the model
set.seed(16)
ridge <- train(
  y ~., data = dataTreino, method = "ridge",
  trControl = trainControl("cv", number = 5),
  tuneLength = 13,
  preProc=c("center","scale","YeoJohnson")
)

# Make predictions

predictionsTrain <- predict(ridge, dataTreino)
predictionsTest <- predict(ridge, dataTeste)
# Model prediction performance

# RMSE e Rsquare do Treino 
data.frame(
  RMSE = RMSE(predictionsTrain, dataTreino$y),
  Rsquare = R2(predictionsTrain, dataTreino$y)
)

# RMSE e Rsquare do Teste
data.frame(
  RMSE = RMSE(predictionsTest, dataTeste$y),
  Rsquare = R2(predictionsTest, dataTeste$y)
)

dfTest = data.frame(dataTeste$y,predictionsTest)
dfTrain = data.frame(dataTreino$y,predictionsTrain)


pdf("L2Train10fold.pdf")
ggplot(dfTreino,aes(dataTreino.y,predictionsTrain)) + geom_point(color="blue") + labs(title="L2-Train: 10-fold",x="Predicted",y="Observed")
dev.off()
pdf("L2Test10fold.pdf")
ggplot(dfTeste,aes(dataTeste.y,predictionsTest)) + geom_point(color="blue") + labs(title="L2_Test: 10-fold",x="Predicted",y="Observed")
dev.off()


#                                               Parte 3 - PLS ou PCR

# Build the model
set.seed(16)
ridge <- train(
  y ~., data = dataTreino, method = "pls",
  trControl = trainControl("cv", number = 5),
  tuneLength = 13,
  preProc=c("center","scale","YeoJohnson")
)

# Make predictions

predictionsTrain <- predict(ridge, dataTreino)
predictionsTest <- predict(ridge, dataTeste)
# Model prediction performance

# RMSE e Rsquare do Treino 
data.frame(
  RMSE = RMSE(predictionsTrain, dataTreino$y),
  Rsquare = R2(predictionsTrain, dataTreino$y)
)

# RMSE e Rsquare do Teste
data.frame(
  RMSE = RMSE(predictionsTest, dataTeste$y),
  Rsquare = R2(predictionsTest, dataTeste$y)
)

dfTest = data.frame(dataTeste$y,predictionsTest)
dfTrain = data.frame(dataTreino$y,predictionsTrain)


pdf("PLSTrain10fold.pdf")
ggplot(dfTreino,aes(dataTreino.y,predictionsTrain)) + geom_point(color="red") + labs(title="PLS-Train: 10-fold",x="Predicted",y="Observed")
dev.off()
pdf("PLSTest10fold.pdf")
ggplot(dfTeste,aes(dataTeste.y,predictionsTest)) + geom_point(color="red") + labs(title="PLS-Test: 10-fold",x="Predicted",y="Observed")
dev.off()

# referencias
# http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/153-penalized-regression-essentials-ridge-lasso-elastic-net/#ridge-regression
# http://ricardoscr.github.io/como-usar-ridge-e-lasso-no-r.html