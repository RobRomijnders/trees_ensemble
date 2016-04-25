# _____ Initialize data ______
# Author: Mareva Brixy
# # Define training set and test set
# train_data = read.csv(file="ModelingTrain_train.csv")
# test_data = read.csv(file="ModelingTrain_test.csv")

# Read the data from the CSV file
data = read.csv(file="ModelingTrain.csv")

# Define the total number of samples
total_nb_samples = length(data[,1])

# Define the number of covariates
nb_covariates = length(data[1,])

# Drop the first column (ID)
data = data[,-1]

# Split the data into a training set and a test set
train_idx = sample(1:total_nb_samples, 0.75 * total_nb_samples)
train_data = data[train_idx,]
test_data = data[-train_idx,]
train_nb_samples = length(train_data[,1])
test_nb_samples = length(test_data[,1])

# Data as categorical data (only integer values)
for (i in 1:nb_covariates){
  train_data[,i] = as.factor(train_data[,i])
  test_data[,i] = as.factor(test_data[,i])
}

# For reproductibility
set.seed(4240)

# _____ Random Forest algorithm ______

# Import the library 'randomForest'
library(randomForest)

# Define the random forest : y is the variable of interest
random_forest = randomForest( y~., data = train_data,
                              ntry = 22, ntree = 100, importance = TRUE)

# Variable importance : assess the importance of the predictors
varImpPlot(random_forest)

# Make predictions on the test set
random_forest_prediction = predict(object = random_forest,
                                   newdata = test_data[, -1], type = "prob")

# Take the highest probability for each sample
random_forest_prediction_final = rep(0, test_nb_samples)
for (i in 1:test_nb_samples) {
  random_forest_prediction_final[i] = which.max(random_forest_prediction[i,]) - 1
}

# Calculate the accuracy
random_forest_accuracy = mean(random_forest_prediction_final == test_data[,1])
