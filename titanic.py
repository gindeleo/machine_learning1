#!/usr/bin/python

#Machine Learning test
#from https://www.kaggle.com/c/titanic

#   **** TASK ****

#	In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive.
#	In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data from csv files
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

# Plots
#plot to get an idea of age data and whether median is a reasonable pick for replacing missing data
# plt.title("median = " + str(test["Age"].median()) + " mean = " + str(test["Age"].mean()))
# plt.xlabel('Age')
# plt.ylabel('frequency')
# test["Age"].plot(kind="hist", bins=16)
# plt.show()

test["Age"] = test["Age"].fillna(test["Age"].median()) #fill missing values with median value
train["Age"] = train["Age"].fillna(train["Age"].median())

#plot for agae data after clean up
# plt.title("median = " + str(test["Age"].median()) + " mean = " + str(test["Age"].mean()))
# plt.xlabel('Age')
# plt.ylabel('frequency')
# test["Age"].plot(kind="hist", bins=16)
# plt.show()

#print out info on train set
print train.head(10)
print train.describe()

#clean up data
test["Age"] = test["Age"].fillna(test["Age"].median()) #fill missing values with median value
train["Age"] = train["Age"].fillna(train["Age"].median())

test["Embarked"]=test["Embarked"].fillna("S")
train["Embarked"]=train["Embarked"].fillna("S")

#make entries numerical for Sex and Embarked
test.loc[test["Sex"] == "male", "Sex"] = 0 ; test.loc[test["Sex"] == "female", "Sex"] = 1
train.loc[train["Sex"] == "male", "Sex"] = 0 ; train.loc[train["Sex"] == "female", "Sex"] = 1

test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2




