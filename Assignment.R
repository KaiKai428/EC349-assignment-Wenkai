rm(list=ls())

library(ggplot2)
library(GGally)
library(caret)
library(dplyr)
library(tidytext)
library(rpart)
library(rpart.plot)
library(glmnet)


# View data structure and summary

## View the first few rows of the dataset
head(review_data_small)

## View the structure of the dataset
str(review_data_small)

## Get summary statistics for each column
summary(review_data_small)

# Exploratory data analysis

# Data cleaning

## Handling Missing Values
### Check for missing values in each column
sum(is.na(review_data_small))

## Consistency Checks
### Convert all text to lower case for consistency
review_data_small$text <- tolower(review_data_small$text)

## Duplicate Data Check
### # Identify duplicate rows
review_data_small[duplicated(review_data_small),]

## Data Distribution Analysis

### Histogram for the distribution of star ratings
hist(review_data_small$stars, main="Distribution of Stars", xlab="Stars")

### Boxplot for the distribution of 'useful' votes
boxplot(review_data_small$useful, main="Boxplot of Useful Votes", ylab="Number of Useful Votes")

### Boxplot for the distribution of 'funny' votes
boxplot(review_data_small$funny, main="Boxplot of Funny Votes", ylab="Number of Funny Votes")

### Boxplot for the distribution of 'cool' votes
boxplot(review_data_small$cool, main="Boxplot of Cool Votes", ylab="Number of Cool Votes")

## Correlation Analysis
# Scatterplot matrix for stars, useful, funny, and cool votes
pairs(review_data_small[, c("stars", "useful", "funny", "cool")], main="Scatterplot Matrix")

## Text Content Analysis

### Adding a new column for text length
review_data_small$text_length <- nchar(review_data_small$text)

### Histogram for the distribution of review text length
hist(review_data_small$text_length, main="Distribution of Review Text Length", xlab="Text Length")


## Time Trend Analysis
### Converting 'date' column to Date type
review_data_small$date <- as.Date(review_data_small$date)

### Line plot for the number of reviews over time
plot(table(review_data_small$date), type="l", main="Number of Reviews Over Time", xlab="Date", ylab="Number of Reviews")

# Model building
## Feature Engineering

### Convert positive and negative sentiment into numerical scores
### Positive emotions are assigned a value of 1, and negative emotions are assigned a value of -1.
sentiments <- get_sentiments("bing") %>%
  mutate(score = ifelse(sentiment == "positive", 1, -1))

### Sentiment analysis of review text
sentiment_scores <- review_data_small %>%
  unnest_tokens(word, text) %>%
  inner_join(sentiments, by = "word") %>%
  group_by(review_id) %>%
  summarize(sentiment_score = sum(score)) 

sentiment_scores <- na.omit(sentiment_scores)
review_data_small <- left_join(review_data_small, sentiment_scores, by = "review_id")


### Aggregate user data
### Calculate average stars given by each user
avg_user_rating <- review_data_small %>%
  group_by(user_id) %>%
  summarize(avg_stars = mean(stars, na.rm = TRUE))

### Merge the user's average rating back into the training set
review_data_small <- merge(review_data_small, avg_user_rating, by = "user_id")

### Count the number of comments per user
user_review_count <- review_data_small %>%
  group_by(user_id) %>%
  summarize(review_count = n())

### Merge the user's number of comments back into the training set
review_data_small <- merge(review_data_small, user_review_count, by = "user_id")


### Aggregate business data
### Calculate average stars received by each business
avg_business_rating <- review_data_small %>%
  group_by(business_id) %>%
  summarize(avg_stars = mean(stars, na.rm = TRUE))

### Merge the business's average rating back into the training set
review_data_small <- merge(review_data_small, avg_business_rating, by = "business_id")

### Count the number of reviews each business received
business_review_count <- review_data_small %>%
  group_by(business_id) %>%
  summarize(review_count = n())

### Merge the business's number of reviews received back into the training set
review_data_small <- merge(review_data_small, business_review_count, by = "business_id")

review_data_small <- na.omit(review_data_small)

## Data Splitting
### Set a seed for reproducibility
set.seed(1)

### Create indices for the training set with an 80% split
trainIndex <- createDataPartition(review_data_small$stars, p = .80, 
                                  list = FALSE, 
                                  times = 1)

### Subset the data to create the training set
train_set <- review_data_small[trainIndex, ]

### Subset the data to create the test set
test_set <- review_data_small[-trainIndex, ]


### Verify the size of the training and test sets
nrow(train_set) # Size of the training set
nrow(test_set)  # Size of the test set


## Model Selection

### Use linear regression as Baseline model
### Set up cross-validation for model training
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,  # Number of folds in cross-validation
                           repeats = 3)  # Number of repeats of CV

### Train a linear regression model using the engineered features
set.seed(1)  # Setting seed for reproducibility

lm_model <- train(stars ~ sentiment_score + avg_stars.x + avg_stars.y + review_count.x + review_count.y,
                  data = train_set, 
                  method = "lm",
                  trControl = fitControl)

# Output the summary of the linear regression model
summary(lm_model)


### Decision tree model

### Training a decision tree model
decision_tree_model <- rpart(stars ~ sentiment_score + avg_stars.x + avg_stars.y + review_count.x + review_count.y,
                             data = train_set, 
                             method = "anova")

### Print the summary of the decision tree model
summary(decision_tree_model)

### Visualizing the tree
rpart.plot(decision_tree_model)


## Lasso model

# Preparing the data for Lasso regression
# Creating a model matrix for the predictors
x <- model.matrix(stars ~ sentiment_score + avg_stars.x + avg_stars.y + review_count.x + review_count.y - 1, data = train_set)

# Extracting the response variable 'stars'
y <- train_set$stars

# Setting seed for reproducibility
set.seed(1)
# Fitting the Lasso model
lasso_model <- glmnet(x, y, alpha = 1)  # alpha = 1 indicates Lasso regression

# Printing the coefficients of the model
print(coef(lasso_model))

# Plotting the coefficients against log(lambda) for model visualization
plot(lasso_model, xvar = "lambda", label = TRUE)

## ModelEvaluation

# Predictions
linear_predictions <- predict(lm_model, newdata = test_set)
decision_tree_predictions <- predict(decision_tree_model, newdata = test_set)
x_test <- model.matrix(~ sentiment_score + avg_stars.x + avg_stars.y + review_count.x + review_count.y - 1, data = test_set)

lasso_predictions <- predict(lasso_model, newx = x_test)


# Calculate MSE for each model
linear_mse <- mean((test_set$stars - linear_predictions) ^ 2)
decision_tree_mse <- mean((test_set$stars - decision_tree_predictions) ^ 2)
lasso_mse <- mean((test_set$stars - lasso_predictions) ^ 2)

# Print the MSE of each model
print(paste("Linear Regression MSE:", linear_mse))
print(paste("Decision Tree MSE:", decision_tree_mse))
print(paste("Lasso Regression MSE:", lasso_mse))




