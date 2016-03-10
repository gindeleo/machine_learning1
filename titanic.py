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

from sklearn.ensemble import RandomForestClassifier 

import argparse
import sys,os,csv


#Parse command line arguments
parser = argparse.ArgumentParser(description='Predict survivors of Titanic using machine learning')
parser.add_argument('--method', '-m', type=str,  help="Select machine learning method (linear, forest) (default=linear)", default='linear', dest="ml_method_input") 
args = parser.parse_args()


#read data from csv files
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

#print out info on train set
# print train.head(10)
# print train.describe()


#clean up data
test["Age"] = test["Age"].fillna(test["Age"].median()) #fill missing values with median value
train["Age"] = train["Age"].fillna(train["Age"].median())

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

# Fare

# only for test_df, since there is a missing "Fare" values
test["Fare"].fillna(test["Fare"].median(), inplace=True)



#Random Forest machine learning
def MlForest(train, test):
	""" Random Forest tree ensemble method """

	test = test.drop(['Cabin','Name','Ticket'], axis=1)
	train = train.drop(['Cabin','Name','Ticket'], axis=1)

	train_data=train.values
	test_data=test.values
	train_data.astype(float); test_data.astype(float)

	np.set_printoptions(threshold=np.nan)
	
	forest = RandomForestClassifier(n_estimators =400,criterion='entropy')
	forest.fit(train_data[:,[0,2,3,4,5,6,7,8]], train_data[:,1].astype(float)) #select all input
	result=forest.predict(test_data[:,[0,1,2,3,4,5,6,7]])

	return result


def ValidateResult(result,compare_list):
	"""Compare results to results from Kaggle (genderclass.csv)"""
	compare_df=pd.read_csv(compare_list)

	accuracy=sum(result==compare_df["Survived"].values)/np.float(np.size(result))
	
	return accuracy


def MlLinear(train, test):
	""" LinearRegression method """

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
	print("cross-validation of Linear regression for train set accuracy: " + str(accuracy))

	# Initialize our algorithm
	alg = LogisticRegression(random_state=1)
	# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
	scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)
	# Take the mean of the scores (because we have one for each fold)
	print("cross-validation of Logistic Regression for train set accuracy: " +str(scores.mean()))


	# Train the algorithm using all the training data
	alg.fit(train[predictors], train["Survived"])

	# Make predictions using the test set.
	predictions = alg.predict(test[predictors])

	return predictions


#select method from command line arguments
ml_methods= {"Linear": "linear", "linear": "linear", "Forest": "forest", "forest": "forest"}
if args.ml_method_input in ml_methods:
	ml_method=ml_methods[args.ml_method_input]
	print "\nMachine learning method: " , ml_method ,'\n'
else:
	print "\n" "'" ,args.ml_method_input, "'","is not a known method. Choose linear or forest"
	sys.exit()

#predict
"Predicting...\n"	
if ml_method == "forest":
	try:
		result = MlForest(train, test)
		print ml_method, "finished"

	except:
		print "Error in prediction: \n"
		raise; sys.exit()

elif ml_method == "linear":
	try:
		result = MlLinear(train, test)
		print ml_method, "finished"

	except:
		print "Error in prediction: \n"
		raise; sys.exit()

predictions_file = open("predicted.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(test["PassengerId"], result))
predictions_file.close()

#validate results
accuracy= ValidateResult(result,'genderclassmodel.csv')

print "\nOverlap with reference set from Kaggle: ", accuracy



