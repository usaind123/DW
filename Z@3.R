# Topic: Accounting Fraud
# The purpose of this project is to predict potential accounting fraud in US public 
#companies. The data is from Bao et al.(2020) article.

#Q1 Importing the CSV File.
data_1 <- read.csv("C:/Users/Dell/Downloads/AccountingFraud.csv")

#Q2 Data preparation
sapply(data_1, function(x) sum(is.na(x)))
# A few variables have missing values.Dropping missing values
data_1 <- na.omit(pd)

library(dplyr)

#Q3 Turn the variable ‘fyear’ into a categorical variable.
train.data$fyear <- factor(train.data$fyear)
test.data$fyear <- factor(test.data$fyear)
class(train.data$fyear)
class(test.data$fyear)

#Q4 Partitioning the data into training and testing datasets
train.data <- filter(data_1,fyear < 2011)
print(train.data$fyear)

test.data <- filter(data_1,fyear >= 2011)
print(test.data$fyear)

#Q5 Building a logistic regression model using training data
logistic_Model <- glm(misstate ~ ., data = train.data, family = binomial)

#Q6 Summary results of the logistic regression model
summary(logistic_Model)

#Q7 Interpret the coefficient of variable ‘reoa’ in terms of its impact on odds. Is it significant? 
#How will the odds of being fraud change when reoa changes?

# As the variable 'reoa' has a coefficient estimate of 0.2119 with a standard error of 0.04122. 
#This suggests that an increase in the value of 'reoa' leads to an increase in the log-odds of 
#misstatement, with a one-unit increase in 'reoa' resulting in a 0.2119 increase in the log-odds 
#of misstatement. 

#Adding the test data level
logistic_Model$xlevels <- union(logistic_Model$xlevels[["fyear"]], levels(test.data$fyear))

#Q8 Use predict() function to get the predicted probabilities for the validation data set.
predict_prob <- predict(logistic_Model, newdata = test.data, type = "response")

#Q9 Make the classification based on probabilities.

#If the predicted probability is greater than 0.5, then the observation will be classified as 'fraud', 
#else'non-fraud'.

predict_class <- ifelse(predict_prob > 0.5, "fraud", "non-fraud")
print(predict_class)

#Q10 Generate the confusion matrix. What is the percentage of accurate prediction?
actualclass <- test.data$misstate
print(actualclass)
confusion_matrix <- table(predict_class, actualclass)
print(confusion_matrix)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(accuracy)