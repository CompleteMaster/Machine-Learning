#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 02:34:11 2021

@author: jorgecompleto
"""

# Regression


# Importing library Pandas

import pandas as pd


# Importing the dataset

dataset = pd.read_excel('Folds5x2_pp.xlsx')

# Independent variables

X = dataset.iloc[:, :-1].values


# Dependent variable

Y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = 0)

# Fitting Gradient Boosting to the Training set
from sklearn.ensemble import GradientBoostingRegressor 

regressor = GradientBoostingRegressor()

regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

