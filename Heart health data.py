# To Build a logistic model to predict whether a person seeks medical treatment in 2 days or less (“1”) or takes longer than 2 days to seek medical treatment (“0”),


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_excel("heart-health-data.xls")
data

#Creating dummy data

data_dummy = pd.get_dummies(data)
data_dummy.dropna(inplace=True)
data_dummy

data_dummy["delaydays_2or_less"] = np.where(data_dummy["delaydays"] <= 2, 1, 0)

X = data_dummy.drop(["ID", "delaydays", "delaydays_2or_less"], axis=1)
y = data_dummy["delaydays_2or_less"]

#training the data set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

X = sm.add_constant(X)

logistic_model = sm.Logit(y_train, X_train).fit()

print(logistic_model.summary())

# Print the coefficients in descending order 
coef = pd.DataFrame({'coef': model.coef_[0], 'variable': X_train.columns})
coef = coef.sort_values(by='coef', ascending=False)
print(coef)

# To modify the logistic regression model to predict whether a person seeks medical treatment on or less than the cohort average delay days (“1”) or takes longer than the average number of days to seek medical treatment (“0”), """

#First, we can calculate the cohort average delay days
avg_delay_days = data_dummy['delaydays'].mean()
print(avg_delay_days)

#creating a binary outcome variable
data_dummy['delaydays_outcome'] = (data_dummy['delaydays'] <= avg_delay_days).astype(int)
data_dummy['delaydays_outcome']

import statsmodels.api as sm

X = data_dummy[['Age', 'Gender', 'Ethnicity', 'Marital', 'Livewith', 'Education', 'palpitations', 'orthopnea', 'chestpain', 'nausea', 'cough', 'fatigue', 'dyspnea', 'edema', 'PND', 'tightshoes', 'weightgain', 'DOE']]
y = data_dummy['delaydays_outcome']

X = sm.add_constant(X)
log_reg_avg = sm.Logit(y, X).fit()

log_reg_avg.summary()



# To build a logistic regression model to predict whether a person seeks medical treatment on or less than 1 day (“1”) or takes longer than 1 day to seek medical treatment (“0”):"""

data_dummy['early_treatment'] = (data_dummy['delaydays'] <= 1).astype(int)

# Create a list of predictor variables
predictors = ['Age', 'Gender', 'Ethnicity', 'Marital', 'Livewith', 'Education', 
              'palpitations', 'orthopnea', 'chestpain', 'nausea', 'cough', 'fatigue', 
              'dyspnea', 'edema', 'PND', 'tightshoes', 'weightgain', 'DOE']

subset_data = data_dummy[predictors]             

# Fit the logistic regression model
logit_model = sm.Logit(data_dummy['early_treatment'], data_dummy[predictors]).fit()

# Print the summary table of the model
print(logit_model.summary())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the logistic model on the training set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict the outcome using the testing set
y_pred = logreg.predict(X_test)

# Evaluate the performance of the model
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

