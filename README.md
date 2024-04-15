# Gannah-s-Assignment-
This is my first PR assignment 



import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
'hours-per-week', 'native-country', 'income']

# load the data
data= pd.read_csv("adult.data", names= columns , na_values=['?'], sep=', ')
data
# the testing data
#we'll skip the first row to remove the 1x3 cross validator
testing_data = pd.read_csv("adult.test", names= columns, skiprows=1, na_values=['?'], sep=', ')
testing_data

# checking missing values

print("missing values of the training set\n", data.isnull().sum())
print("sum of null values of the training set : " ,data.isnull().sum().sum())

print("\n")

print("missing values of the testing set\n", testing_data.isnull().sum())
print("sum of null values of the testing set : " ,testing_data.isnull().sum().sum())

# drop na 

data = data.dropna()
testing_data = testing_data.dropna()

print("missing values of the training set after dropping the missing values : \n", data.isnull().sum())
print("sum of null values of the training set after dropping the missing values : " ,data.isnull().sum().sum())

print("\n")

print("missing values of the testing set after dropping the missing values : \n", testing_data.isnull().sum())
print("sum of null values of the testing set after dropping the missing values  : " ,testing_data.isnull().sum().sum())
# split the data into the inputs and the target variable (the income)
x_train = data.drop("income" , axis=1)
y_train = data['income']
x_test = testing_data.drop("income" , axis=1)
y_test = testing_data['income']
We'll split the target variable from the set we want to apply the one hot conding method (turns categorical values into binary values) to improve the prediction accuracy.

Afterwards there'll be slight issue that we'll bet met with where Holland-Netherlands is found in the training set and not in the testing set thus there'll be an extra column in the training that isn't found in the testing which'll give an error.

To fix that we'll need to concatinate the training set with the testing set so that there'll be Holland-Netherlands in both sets.

And then we'll split them again by taking the data from 0 to the length of the x_train and the test starts from the length of the x_train till the end.
new_data = pd.concat([x_train,x_test],axis=0)

encoding = pd.get_dummies(new_data,columns=new_data.select_dtypes(include=["object"]).columns)

encoded_x_train = encoding[:len(x_train)]
encoded_x_test = encoding[len(x_train):]


# instead of false & true to 0's to 1's
binary_y_train = (y_train == '>50K').astype(int)
binary_y_test = (y_test == '>50K.').astype(int)


print("y_train: \n" ,y_train)
print("binary_y_train : \n", binary_y_train)
print("y_test: \n" ,y_test)
print("binary_y_test : \n", binary_y_test)

naive_bayes  = GaussianNB()
naive_bayes.fit(encoded_x_train, binary_y_train)
# predicting

y_pred = naive_bayes.predict(encoded_x_test)
y_pred

confusion matrix

                                   Actual 
                              yes          no 
                       yes    TP           FP
                    predicted
                       no     FN           TN      
# confusion matrix
tn,fp,fn,tp = confusion_matrix(binary_y_test,y_pred).ravel()

# sensitivity
sensitivity = tp/(tp+fn)
print("sensitivity = ",sensitivity)

# specificity 
specificity = tn/(tn+fp)
print("specificity = ",specificity)

#Posterior Probability
posterior_prob = naive_bayes.predict_proba(encoded_x_test)
print("Posterior probability of making a yearly income of >=50K is =  ",posterior_prob[:,1])

