#!/usr/bin/python

#Machine Learning test
#from https://www.kaggle.com/c/titanic

#   **** TASK ****

#	In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive.
#	In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import random as rnd
from sklearn import cross_validation


#read data from csv files
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

#print out info on train set
# print train.head(10)
# print train.describe()

#clean up data


test["Embarked"]=test["Embarked"].fillna(1)
train["Embarked"]=train["Embarked"].fillna(1)

#make entries numerical for Sex and Embarked
test.loc[test["Sex"] == "male", "Sex"] = 0 ; test.loc[test["Sex"] == "female", "Sex"] = 1
train.loc[train["Sex"] == "male", "Sex"] = 0 ; train.loc[train["Sex"] == "female", "Sex"] = 1

test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

test = test.drop(['Cabin','Name','Ticket'], axis=1)
train = train.drop(['Cabin','Name','Ticket'], axis=1)

# Fare

# only for test_df, since there is a missing "Fare" values
test["Fare"].fillna(test["Fare"].median(), inplace=True)


# Age
#fill missing values with median
# # test_rand_age = test["Age"].median()
# # train_rand_age = train["Age"].median()
#fill missing values with N(median,std) 
test_rand_age = np.random.normal(test["Age"].mean(),2*np.std(test["Age"]), size= None)
train_rand_age = np.random.normal(train["Age"].mean(), 2*np.std(train["Age"]), size= None)
#No appreciable difference between these methods of filling the missing age values

test["Age"] = test["Age"].fillna(test_rand_age) 
train["Age"] = train["Age"].fillna(train_rand_age)
# convert from float to int
train['Age'] = train['Age'].astype(int)
test['Age'] = test['Age'].astype(int)


# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the train dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.

kf = KFold(train.shape[0], n_folds=3, random_state=1)

predictions = []
for train1, test1 in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (train[predictors].iloc[train1,:])
    # The target we're using to train the algorithm.
    train_target = train["Survived"].iloc[train1]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(train[predictors].iloc[test1,:])
    predictions.append(test_predictions)
    

predictions = np.concatenate(predictions, axis=0)

predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

accuracy = sum(train["Survived"] == predictions)/len(train["Survived"])
print("cross-validation ofLinear regression for train set accuracy: " + str(accuracy))

# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print("cross-validation of LogisticRegression for train set accuracy: " +str(scores.mean()))


# Train the algorithm using all the training data
alg.fit(train[predictors], train["Survived"])

# Make predictions using the test set.
predictions = alg.predict(test[predictors])


np.savetxt("predictions.csv", predictions, delimiter=",")



