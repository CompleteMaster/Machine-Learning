#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 01:38:14 2021

@author: jorgecompleto
"""

# Classification


# Importing library Pandas

import pandas as pd


# Importing the dataset

dataset = pd.read_csv('breast-cancer-wisconsin.csv')

# Independent variables

X = dataset.iloc[:, 1:-1].values


# Dependent variable

Y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = 0)

# Fitting Gradient Boosting to the Training set
from sklearn.ensemble import GradientBoostingClassifier 

classifier = GradientBoostingClassifier()

classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Compute the Accuracy of the Model

# 1 - (5/137) = 0.9635036496350365





