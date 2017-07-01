#Download Data Files:
#spamdata.csv:  
#spamnames.csv: 

#Load the two files spamdata.csv and spamnames.csv into R:

#Load the values
spamdata<- read.csv("spamdata.csv",header=FALSE,sep=";")

#Load the column names
spamnames<- read.csv("spamnames.csv",header=FALSE,sep=";")


#Set the column names of the dataframe spamdata
names(spamdata) <- sapply((1:nrow(spamnames)),function(i) toString(spamnames[i,1]))


#Converting into factors/categorical values so that we can use SVM for classification
spamdata$y <- factor(spamdata$y)


#Creating a sample dataset of 1000 rows
sample <- spamdata[sample(nrow(spamdata), 1000),]



#Loading the packages

#Loading caret


install.packages("caret", dependencies = c("Depends", "Suggests"))

require(caret)

#Loading kernlab

install.packages("kernlab", dependencies = c("Depends", "Suggests"))

require(kernlab)

#Loading doParallel
install.packages("doParallel", dependencies = c("Depends", "Suggests"))

require(doParallel)

#Split the data in two parts trainData and testData
trainIndex <- createDataPartition(sample$y, p = .8, list = FALSE, times = 1)
trainData <- sample[ trainIndex,]
testData <- sample[-trainIndex,]

#set up multicore environment
registerDoParallel(cores=5)


#Create the SVM model:
# computing optimal value for the  tuning parameter
sigDist <- sigest(y ~ ., data = trainData, frac = 1)
### creating a grid of two tuning parameters, .sigma comes from the earlier line. we are trying to find best value of .C
svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-2:7))

svm_model<- train(y ~ .,
                  data = trainData,
                  method = "svmRadial",
                  preProc = c("center", "scale"),
                  tuneGrid = svmTuneGrid,
                  trControl = trainControl(method = "repeatedcv", repeats = 5, 
                                           classProbs =  FALSE))


#Apply the model on testData for evaluation
predict_spam <- predict(svm_model,testData[,1:57])

#Determine the accuracy of model

accuracy <- confusionMatrix(predict_spam, testData$y)

#write the output to Result.csv

write.csv(predict_spam, file = "Result.csv")
