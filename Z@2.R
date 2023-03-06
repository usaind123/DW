#Q-1 Import ToyotaCorolla.csv file
data <- read.csv("C:/Users/Dell/Downloads/ToyotaCorolla.csv")

#Q-2 The data set includes sale prices and vehicle characteristics of 1436 used Toyota Corollas. 
View(data)

#Q-3 Explore the relationship between variable KM and Price using a plot. What does the relationship between KM and Price look like? 
plot(data$KM,data$Price)

#Q-4 Check the data types of “MetColor” and “Automatic”. Do you think they are appropriate data types for the variable? Why? Transform them if needed.
class(data$MetColor)
class(data$Automatic)
#The variables are of integer data type but as they have categorical data,
catg1 <- as.factor(data$MetColor)
catg2 <- as.factor(data$Automatic)
class(catg1)
class(catg2)

#Q-5 Unique values of Doors 
unique(data$Doors)
# how many of them?
length(unique(data$Doors))
#data type Doors
class(data$Doors)
#we can check for required number of doors and the other relevant information from the other variables.

#Q-6 Run a linear regression using all predictor variables.
data.lm <- lm(Price ~ .,data = data)

#Q-7 Which variables have significant impact on the sale price of used Toyota Corolla?
summary(data.lm)
#the summary depicts that all the variables("Age","KM","Fuel_Type","HP","MetColor","Automatic","Doors") have a major impact on Price variable for a used Toyota Corolla.

#Q-8 What is the confidence interval of the coefficients of “Age” and “Automatic”? 
confint(data.lm)
#The older vehicle has lower worth, according to the negative coefficients for age. According to the positive coefficients for Automatic, vehicles with automatic depreciate less quickly than other vehicles

#Q-9 How well does the model fit the data?
plot(data.lm)

#Q-10 Make predictions of sale price for the sample using confidence and prediction intervals.
p1 <- predict(data.lm, interval = "confidence")
p2 <- predict(data.lm, interval = "prediction")
