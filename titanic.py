#!/usr/bin/python

#Machine Learning test
#from https://www.kaggle.com/c/titanic

#   **** TASK ****

#	In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive.
#	In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

#drop unused columns
test = test.drop(['Cabin','Name','Ticket'], axis=1)
train = train.drop(['Cabin','Name','Ticket'], axis=1)

test["Fare"].fillna(test["Fare"].median(), inplace=True)


#select method from command line arguments
ml_methods= {"Linear": "linear", "linear": "linear", "Forest": "forest", "forest": "forest"}
if args.ml_method_input in ml_methods:
	ml_method=ml_methods[args.ml_method_input]
	print "\nMachine learning method: " , ml_method ,'\n'
else:
	print "\n" "'" ,args.ml_method_input, "'","is not a known method. Choose linear or forest"
	sys.exit()


#Define prediction methods

#Random Forest machine learning
def MlForest(train, test):
	""" Random Forest tree ensemble method """

	train_data=train.values
	test_data=test.values
	train_data.astype(float); test_data.astype(float)

	np.set_printoptions(threshold=np.nan)
	
	forest = RandomForestClassifier(n_estimators =200,criterion='entropy')
	forest.fit(train_data[:,[0,2,3,4,5,6,7,8]], train_data[:,1].astype(float)) #select all input
	result=forest.predict(test_data[:,[0,1,2,3,4,5,6,7]])

	return result


def ValidateResult(result,compare_list):
	"""Compare results to results from Kaggle (genderclass.csv)"""
	compare_df=pd.read_csv(compare_list)

	accuracy=sum(result==compare_df["Survived"].values)/np.float(np.size(result))
	
	return accuracy

#predict
"Predicting...\n"	
if ml_method == "forest":
	try:
		result = MlForest(train, test)
		print ml_method, "finished"

	except:
		print "Error in prediction: \n"
		raise; sys.exit()


#elif ML_Method == "linear":


predictions_file = open("predicted.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(test["PassengerId"], result))
predictions_file.close()

#validate results
accuracy= ValidateResult(result,'genderclassmodel.csv')

print "\nOverlap with reference set from Kaggle: ", accuracy







