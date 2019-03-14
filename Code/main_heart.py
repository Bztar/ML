# -*- coding: utf-8 -*-
"""
https://scikit-learn.org/stable/
https://github.com/sujoyde89/driven-data-kernels
"""
from data_class import dataObject
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 40)

# Create an instance of training values
values = dataObject('train_values.csv')

# Create an instance of training labels
labels = dataObject('train_labels.csv')

# Create testdata
test = dataObject('test_values.csv')

# Read data
values.read_data('patient_id')
labels.read_data('patient_id')
test.read_data('patient_id')

# Clean the data
values.clean()
test.clean()

# Split data into train/test set
values.train_test_set(labels.df)

# Send training data to select feature importance
values.feature_selection(values.X_train, values.y_train)

# Show feature importance
values.show_features()

# Import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Import cross val and metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

# Make a list of Classifiers
classifiers = [LogisticRegression(),
               KNeighborsClassifier(n_neighbors=3),
               GaussianNB(),
               RandomForestClassifier(n_estimators=1000, random_state=8)]

# Cross val scores for the different clf
for clf in classifiers:
    name = clf.__class__.__name__
    Accuracy = cross_val_score(clf, 
                               values.X_train, 
                               values.y_train, 
                               scoring='accuracy',
                               cv=5).mean()
    
    LogLoss = cross_val_score(clf, 
                               values.X_train, 
                               values.y_train, 
                               scoring='neg_log_loss',
                               cv=5).mean()
    
    ROC_AUC = cross_val_score(clf, 
                               values.X_train, 
                               values.y_train, 
                               scoring='roc_auc',
                               cv=5).mean()
    
    print('='*30)
    print(name)
    
    print('**Results**')
    print('Accuracy: {}'.format(Accuracy))
    print('LogLoss: {}'.format(LogLoss))
    print('ROC_AUC: {}'.format(ROC_AUC))

print('='*30)

