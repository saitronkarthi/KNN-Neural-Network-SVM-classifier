# Last Name: Rajamani
# First Name: Karthikeyan
# UTA Id: 1001267157
#KNN, Neural Network,SVM Classifier
library(ggvis)
library(plyr)
library(class)
library(nnet)
library(e1071) 
library("neuralnet")
InData<-read.csv("C:\\Users\\Admin\\Desktop\\11.Data Mining-(CSE5334)\\Assignments\\Assignment7\\processed_cleveland_heartdisease.csv");
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) 
  }
# Replaced ? in dataset with NA
myDataIn<-na.omit(InData)
# Removing the heartdisease the predicted attribute
myData<-as.data.frame(lapply(myDataIn[,1:13], normalize))
# adding back heartdisease after normalization
myData$heartdisease <- myDataIn$heartdisease
# find the correlation between heartdiseas & age
cor(myData$heartdisease,myData$age);
names(myData)
# plot them for inspection
plot( myData$heartdisease~myData$age);
myData %>% ggvis(~chol, ~age, fill = ~heartdisease) %>% layer_points()
folds <- split(myData, cut(sample(1:nrow(myData)),10));
n=length(folds);
KNNacc <- rep(NA,length(folds))
NNETacc <- rep(NA,length(folds))
SVMacc <- rep(NA,length(folds))
#K fold cross validation k=10
for(i in 1:n){
  test<-ldply(folds[i])
  train<-ldply(folds[-i])
  #KNN Model
  KNN_Train_Model=knn(train=train[,2:14],test=test[,2:14],cl=train$heartdisease,k=3)
  KNN_confusion.matrix <- table(KNN_Train_Model,test$heartdisease)
  KNNacc[i] <- sum(diag(KNN_confusion.matrix))/sum(KNN_confusion.matrix)
  #NNET_Train_Model=nnet(train[,2:14],train[15], size=10)
  #Neural Network Model
  trainingdata <- cbind(train$age+train$sex+train$cp+train$trestbps+train$chol,train[15])
  colnames(trainingdata) <- c("Input","Output")
  NNET_Train_Model= neuralnet(Output~Input,trainingdata, hidden=10, threshold=0.01)
  NNET_Predict<-compute(NNET_Train_Model,test$age+test$chol+test$chol+test$trestbps)
  NNET_Predict.round <- round(as.numeric(NNET_Predict$net.result))
  NNET_confusion.matrix <- table(test$heartdisease, NNET_Predict.round)
  NNETacc[i]<-sum(diag(NNET_confusion.matrix))/sum(NNET_confusion.matrix)
  #SVM Model
  SVM_Train_Model<-svm(heartdisease~age+sex+cp+trestbps+chol,data=train,type="C-classification")
  RSVM_Predict<-predict(SVM_Train_Model,test[2:10])
  SVM_Predict<-predict(SVM_Train_Model,test[,2:15])
  SVM_confusion_matrix<-table(test$heartdisease,SVM_Predict)
  SVMacc[i]<-sum(diag(SVM_confusion_matrix))/sum(SVM_confusion_matrix)
}
# plot the KNN , Neural Network & SVM Models
plot(test$heartdisease~test$age,KNN_Train_Model)
plot(NNET_Train_Model)
plot(SVM_Predict)

cat ("The Accuracy of KNN Model is", mean(KNNacc))
cat ( "The Accuracy of Neural Network",mean(NNETacc))
cat("The accuracy  of SVM Model is",mean(SVMacc))

