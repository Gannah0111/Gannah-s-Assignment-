# Gannah-s-Assignment-
This is my first PR assignment 

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
'hours-per-week', 'native-country', 'income']
data= pd.read_csv("adult.data", names= columns , na_values=['?'], sep=', ')
data
testing_data = pd.read_csv("adult.test", names= columns, skiprows=1, na_values=['?'], sep=', ')
testing_data
print("missing values of the training set\n", data.isnull().sum())
print("sum of null values of the training set : " ,data.isnull().sum().sum())
print("\n")
print("missing values of the testing set\n", testing_data.isnull().sum())
print("sum of null values of the testing set : " ,testing_data.isnull().sum().sum())
data = data.dropna()
testing_data = testing_data.dropna()
print("missing values of the training set after dropping the missing values : \n", data.isnull().sum())
print("sum of null values of the training set after dropping the missing values : " ,data.isnull().sum().sum())
print("missing values of the testing set after dropping the missing values : \n", testing_data.isnull().sum())
print("sum of null values of the testing set after dropping the missing values  : " ,testing_data.isnull().sum().sum())
x_train = data.drop("income" , axis=1)
y_train = data['income']
x_test = testing_data.drop("income" , axis=1)
y_test = testing_data['income']
new_data = pd.concat([x_train,x_test],axis=0)
encoding = pd.get_dummies(new_data,columns=new_data.select_dtypes(include=["object"]).columns)
encoded_x_train = encoding[:len(x_train)]
encoded_x_test = encoding[len(x_train):]
binary_y_train = (y_train == '>50K').astype(int)
binary_y_test = (y_test == '>50K.').astype(int)
print("y_train: \n" ,y_train)
print("binary_y_train : \n", binary_y_train)
print("y_test: \n" ,y_test)
print("binary_y_test : \n", binary_y_test)
naive_bayes  = GaussianNB()
naive_bayes.fit(encoded_x_train, binary_y_train)
y_pred = naive_bayes.predict(encoded_x_test)
y_pred
tn,fp,fn,tp = confusion_matrix(binary_y_test,y_pred).ravel()
sensitivity = tp/(tp+fn)
print("sensitivity = ",sensitivity)
specificity = tn/(tn+fp)
print("specificity = ",specificity)
posterior_prob = naive_bayes.predict_proba(encoded_x_test)
print("Posterior probability of making a yearly income of >=50K is =  ",posterior_prob[:,1])

