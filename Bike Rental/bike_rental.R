


#install.package(c("dplyr","plyr","data.table","ggplot2","ggplot"))

library("dplyr")
library("plyr")
library("data.table")
library("ggplot2")
library("ggplot")

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

#Box plot for all numeric variables
par(mfrow = c(3,3))

boxplot(bike$casual)

boxplot(bike$temp)

boxplot(bike$hum,xlab="hum")
boxplot(bike$atemp)
boxplot(bike$windspeed)
boxplot(bike$registered)
boxplot(bike$cnt)

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


