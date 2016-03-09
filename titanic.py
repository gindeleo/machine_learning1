#!/usr/bin/python

#Machine Learning test 
#from https://www.kaggle.com/c/titanic

#   **** TASK ****

#	In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. 
#	In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

import numpy as np 
import pandas as pd 


#read data from csv files
test_df = pd.read_csv('test.csv')
train_df = pd.read_csv('train.csv')

print test_df.head(10)
print test_df.describe()

