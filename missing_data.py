# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 23:43:27 2016

@author: cgm
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.cs')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of misssing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = Imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
