# -*- coding: utf-8 -*-
"""
Data Analytics Computing Follow Along: Train Test Split and Overfitting
Spyder version 5.3.3
"""

# Import the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# We will perform regression and classification with multiple attempts to fit linear models
# However, this time around, we will focus on the train test split, or how to split the data
# In addition to that, we will also explore overfitting and why it is a problem in machine learning

# Run the lines below if you do not have the packages
# pip install pandas
# pip install sklearn

# This time, look at the loans dataset, a dataset with information about loans in the U.S.

loan_1 = pd.read_csv('C:/Videos/Loan/loan_train.csv')

loan_2 = pd.read_csv('C:/Videos/Loan/loan_test.csv')

# We can concatenate the data even though it is already split for us

loan_data = pd.concat([loan_1, loan_2])

# See how we concatenate in more detail

loan_1.head()

loan_2.head()

loan_data.head()

# Double check that all the records are there by using shape
# This line returns a boolean value
loan_1.shape[0] + loan_2.shape[0] == loan_data.shape[0]

# Subset the data, we do not need all the variables

loan_sub = loan_data[['Married', 'Education',
           'Applicant_Income', 'Loan_Amount', 'Credit_History', 'Term']]

# Use Loan_Amount as a target for regression and Status for classification


# Check for nulls
loan_sub.isnull().sum()

# Drop the nulls
# inplace = True makes sre that loan_sub will be replaced
loan_sub.dropna(axis = 0, inplace = True)

# Let's start by fitting a linear model on the whole dataset

X = loan_sub['Applicant_Income'].values.reshape(-1, 1)

y = loan_sub['Loan_Amount']

# Import the linear regression model from sklearn

linear_model = LinearRegression()

# Fit the model using applicant income as our X and loan amount as of y
# We are fitting on these two variables for simplicity and to show the plot

linear_model.fit(X, y)

# Then get the score of the model

linear_model.score(X, y)

# Get the predictions for the plot

lm_predictions = linear_model.predict(X)

# Plot the results of the first model

plt.ticklabel_format(style='plain')
plt.xlabel("Applicant Income (Thousands)")
plt.ylabel("Loan Amount(Thousands)")
plt.title("Linear Regression on Applicant Income vs Loan Amount")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X/1000, y/1000, color = "Blue", s = 2)
plt.plot(X/1000, lm_predictions/1000, color = "black", linewidth = 2)
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.show()

# Now we can compare the results with the train test split
# Notice the 0.40, this is how big our test dataset will be

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 350)

linear_model_1 = LinearRegression()

linear_model_1.fit(X_train, y_train)

linear_model.score(X_test, y_test)

lm_predictions_1 = linear_model_1.predict(X_test)

# Plot the results of the second model
# Notice how we are evaluating the model on the test data, and graphing it

plt.ticklabel_format(style='plain')
plt.xlabel("Applicant Income (Thousands)")
plt.ylabel("Loan Amount(Thousands)")
plt.title("Linear Regression on Applicant Income vs Loan Amount")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X_test/1000, y_test/1000, color = "Blue", s = 2)
plt.plot(X_test/1000, lm_predictions_1/1000, color = "black", linewidth = 2)
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.show()

# We can also use the original partition
# Train a third model

linear_model_orig = LinearRegression()

X_2 = loan_1['Applicant_Income'].values.reshape(-1, 1)

y_2 = loan_1['Loan_Amount']

linear_model_orig.fit(X_2, y_2)

X_2_test = loan_2['Applicant_Income'].values.reshape(-1, 1)

y_2_test = loan_2['Loan_Amount']

linear_model.score(X_2_test, y_2_test)

lm_predictions_orig = linear_model_orig.predict(X_2_test)

# Plot the results of the third model

plt.ticklabel_format(style='plain')
plt.xlabel("Applicant Income (Thousands)")
plt.ylabel("Loan Amount(Thousands)")
plt.title("Linear Regression on Applicant Income vs Loan Amount")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X_2_test/1000, y_2_test/1000, color = "Blue", s = 2)
plt.plot(X_2_test/1000, lm_predictions_orig/1000, color = "black", linewidth = 2)
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.show()

# Notice how the data points are different, the regression lines are also somewhat different
print(f" The first model's intercept is: {linear_model.intercept_} and the coefficient is:  {linear_model.coef_}")
print(f" The second model's intercept is: {linear_model_1.intercept_} and the coefficient is:  {linear_model_1.coef_}")
print(f" The third model's intercept is: {linear_model_orig.intercept_} and the coefficient is:  {linear_model_orig.coef_}")

# Even though we are taking the same data, how we train it matters. More so when it comes to the regression line
# If we repeated it by multiple partitions, we would end up with a more robust approach called cross-validation
# This random partition gets even more important if we expand the model to Multiple Linear Regression

linear_model_multi = LinearRegression()

X_m = loan_sub[['Applicant_Income', 'Term', 'Credit_History']]

y_m = loan_sub['Loan_Amount']

linear_model_multi.fit(X_m, y_m)

X_m_test = loan_sub[['Applicant_Income', 'Term', 'Credit_History']]

y_m_test = loan_sub['Loan_Amount']

linear_model_multi.score(X_m_test, y_m_test)

lm_predictions_multi = linear_model_multi.predict(X_m_test)

# Plot the results of the third model

plt.ticklabel_format(style='plain')
plt.xlabel("Applicant Income (Thousands)")
plt.ylabel("Loan Amount(Thousands)")
plt.title("Linear Regression on Applicant Income, Term and Credit History vs Loan Amount")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X_m_test['Applicant_Income']/1000, y_m/1000, color = "Blue", s = 2)
plt.scatter(X_m_test['Applicant_Income']/1000, lm_predictions_multi/1000, color = "black", s = 2)
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.show()

# If we take random observations, we can have a lot of variation on the model
# We train the model using the training data, but evaluate it on the testing data
# A simple analogy is that we are using our model to predict data that is has not seen yet

# Let's take a look at Logistic Regression to reinforce these changes in a classification scenario

log_model = LogisticRegression()

# Train the Logistic Regression Model in the data

y_class = loan_1["Status"]

order = ("No", "Yes")

labelencod = LabelEncoder()

y_class = labelencod.fit_transform(y_class)

log_model.fit(X_2, y_class)

# Make sure that the order is correct, as we are predicting the probability of the classes

log_model.classes_

# For classification, instead of score we will use a confusion matrix

log_predictions = log_model.predict(X_2)

# Use the confusion matrix function
# For the confusion matrix, we will be looking at the top left and bottom right values
# The top left value is the true positives
# The bottom right value is the true negatives

confusion_matrix(y_class, log_predictions)

log_predictions_prob = log_model.predict_proba(X_2)[:, 1]

# Plot the model, the probabilities are in orange

plt.ticklabel_format(style='plain')
plt.xlabel("Applicant Income")
plt.ylabel("Status")
plt.title("Logistic Regression Probabilities Applicant Income vs Status of Loan")

plt.scatter(X_2/1000, y_class)
plt.scatter(X_2/1000, log_predictions_prob)
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.show()

# A value is classified as 1 if the probability is more than 0.5, and 0 if not
# This is know as a cutoff value or a threshold

# Now, let's use the train test split on Logistic Regression

log_model_2 = LogisticRegression()

y_class = loan_1["Status"]

order = ("No", "Yes")

labelencod = LabelEncoder()

y_class = labelencod.fit_transform(y_class)

# Split the data

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, 
                                                            y_class, test_size = 0.40, random_state = 350)

# Train the Logistic Regression Model in the data

log_model_2.fit(X_train_2, y_train_2)

# Make sure that the order is correct, as we are predicting the probability of the classes

log_model_2.classes_

# For classification, instead of score we will use a confusion matrix

log_predictions_2 = log_model_2.predict(X_test_2)

# Use the confusion matrix function
# For the confusion matrix, we will be looking at the top left and bottom right values
# The top left value is the true positives
# The bottom right value is the true negatives

confusion_matrix(y_test_2, log_predictions_2)

log_predictions_prob_2 = log_model_2.predict_proba(X_test_2)[:, 1]

# Plot the model, the probabilities are in orange

plt.ticklabel_format(style='plain')
plt.xlabel("Applicant Income")
plt.ylabel("Status")
plt.title("Logistic Regression Probabilities Applicant Income vs Status of Loan")

plt.scatter(X_test_2/1000, y_test_2)
plt.scatter(X_test_2/1000, log_predictions_prob_2)
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.show()