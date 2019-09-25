


#install.package(c("dplyr","plyr","data.table","ggplot2","ggplot"))

library("dplyr")
library("plyr")
library("data.table")
library("ggplot2")
library("corrplot")
library("rpart")
library("MASS")
library("caTools")
library("Boruta") # For better feature selection
library("caret")
library("randomForest")

# Set working directory

setwd("C:/Users/pdas/Desktop/TDS/Personal/EdWisr/Bike Rental")

# load the dataset to R

bike<- read.csv("day.csv")

#Heading and summary of data

head(bike)

summary(bike) # From this data set season, yr, mnth,holiday,weekday,workingday,weatherlist columns are categorical variables.
              # Remaining Temp,atemp, hum, windspeed,casual,registered are numeric varibale.



# cnt is the target variable

# Finding missing value

sum(is.na(bike)) # There is no missing value is the dataset.

# Get structure of dataset
str(bike)

# convert int to factior for categorical variables.

categorical_variables <- c("mnth","holiday","weekday","workingday","weathersit")
numeric_variable<-c("temp","atemp","hum","windspeed","casual","registered","cnt")

bike[categorical_variables] <- lapply(bike[categorical_variables],factor)

numeric_variable<- bike[numeric_variable]
categorical_variables<-bike[categorical_variables]

#Box plot for all numeric variables to check outliers.

par(mfrow = c(3,3))

boxplot(bike$casual, xlab = "casual")

boxplot(bike$temp,xlab ="temp")

boxplot(bike$hum,xlab="hum")
boxplot(bike$atemp,xlab = "atemp")
boxplot(bike$windspeed,xlab = "windspeed")
boxplot(bike$registered,xlab = "regestered")
boxplot(bike$cnt,xlab= "cnt")

# Density Plot for numerical variables


plot(bike)
par(mfrow = c(3,3))
plot(density(bike$temp)) # This density is mostly same with atemp
plot(density(bike$atemp))
plot(density(bike$hum))
plot(density(bike$windspeed))
plot(density(bike$casual))
plot(density(bike$registered))
plot(density(bike$cnt))

# Corelation plot to understand the relationship of each independent variable with target variable.
corr_num = cor(numeric_variable,method = "s")
corr_num
par(mfrow =c(1,1))
corrplot(corr_num)

# There is good correlation betwn "temp" and "atemp" and also "regester" and "casual". So we can keep one of then to reduce the chances of model over fitting.

# scatterplot to understand relationship betwn categorical variable
plot(categorical_variables)

# Remove outlier from casual variable using a customised function.

remove_outlier<-function(x,na.rm = TRUE)
  { 
    qnt<-quantile(x,probs = c(.25,.75),na.rm)
    H<-1.5*IQR(x,na.rm=na.rm)
    y<-x
    y[x<(qnt[1]-H)]<-NA
    y[x>(qnt[2]+H)]<-NA
    x<-y
    x[!is.na(x)]
}

##########################################

casual<-bike$casual

par(mfrow=c(1,2))

boxplot(bike$casual,horizontal =T, xlab = "casual_before_outlier_removal")

new<-remove_outlier(casual)
boxplot(new,horizontal = T,xlab="casual_after_outlier_removal")

# Features selection

set.seed(111)

colnames(bike)

features_test <- c("season","yr","mnth","holiday","weekday","workingday","weathersit"
              ,"temp","atemp","hum","windspeed","casual","registered","cnt")

boruta<- Boruta(cnt ~.,data= bike,doTrace =2,maxRuns =500)
print(boruta)

plot(boruta)
features <- c("season","yr","mnth","weekday","workingday","weathersit"
              ,"temp","atemp","hum","windspeed","casual","registered","cnt")
features<-bike[features]

features_test<- bike[features_test]

train_index <- createDataPartition(features_test$cnt, p = .75, list = FALSE)

train<- features_test[train_index,]

test<-features_test[-train_index,]

# Decision Tree MODEL

####################################################################

fit = rpart(cnt~.,data = train, method = "anova")

firstpred = predict(fit,test)

#MAPE

MAPE <- function(y, yPred)
  {
    mean(abs((y-yPred)/y))
}

#RMSE
RMSE <- function(y_test,y_predict) {
  
  difference = y_test - y_predict
  root_mean_square = sqrt(mean(difference^2))
  return(root_mean_square)
  
}

MAPE(test$cnt,firstpred)
RMSE(test$cnt,firstpred)

############################################################################
#Random forest model

random_forest <- randomForest(cnt~.,data = train)

second_pred <- predict(random_forest,test)

MAPE(test$cnt,second_pred)
RMSE(test$cnt,second_pred)

# Random Forest model with parameter tuning

random_forest <- randomForest(cnt~.,data = train,mtry =7,ntree=500 ,nodesize =10 ,importance =TRUE)

second_pred <- predict(random_forest,test)

MAPE(test$cnt,second_pred)
RMSE(test$cnt,second_pred)
