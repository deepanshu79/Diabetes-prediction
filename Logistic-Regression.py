# Logistic regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset1 = pd.read_csv('Diabetes_XTrain.csv')
X = dataset1.iloc[:, :].values
dataset2 = pd.read_csv('Diabetes_YTrain.csv')
y = dataset2.iloc[:, 0].values
dataset3 = pd.read_csv('Diabetes_Xtest.csv')
testing = dataset3.iloc[:, :].values


# Data visualization
preg = X[:, 0]
plt.scatter(preg, y)
plt.title('Pregnancies vs Diabetes outcome')
plt.xlabel('Pregnancies')
plt.ylabel('Diabetes outcome')
plt.show()

glu = X[:, 1]
plt.scatter(glu, y)
plt.title('Glucose level vs Diabetes outcome')
plt.xlabel('Glucose level')
plt.ylabel('Diabetes outcome')
plt.show()

bld = X[:, 2]
plt.scatter(bld, y)
plt.title('Blood pressure vs Diabetes outcome')
plt.xlabel('Blood pressure')
plt.ylabel('Diabetes outcome')
plt.show()

skn = X[:, 3]
plt.scatter(skn, y)
plt.title('Skin thickness vs Diabetes outcome')
plt.xlabel('Skin thickness')
plt.ylabel('Diabetes outcome')
plt.show()

ins = X[:, 4]
plt.scatter(ins, y)
plt.title('Insulin vs Diabetes outcome')
plt.xlabel('Insulin')
plt.ylabel('Diabetes outcome')
plt.show()

bmi = X[:, 5]
plt.scatter(bmi, y)
plt.title('BMI vs Diabetes outcome')
plt.xlabel('BMI')
plt.ylabel('Diabetes outcome')
plt.show()

dpf = X[:, 6]
plt.scatter(dpf, y)
plt.title('Diabetes Pedigree Function vs Diabetes outcome')
plt.xlabel('Diabetes Pedigree Function')
plt.ylabel('Diabetes outcome')
plt.show()

age = X[:, 7]
plt.scatter(age, y)
plt.title('Age vs Diabetes outcome')
plt.xlabel('Age')
plt.ylabel('Diabetes outcome')
plt.show()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 8)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
testing = sc.transform(testing)


# Training the logistic regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# R-Squared and Adjusted R-Squared values
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_pred)
print(R2)
n = int(576*0.05)
p = 8
AR2 = 1-(1-R2)*(n-1)/(n-p-1)
print(AR2)

# Test file prediction
prediction = classifier.predict(testing)