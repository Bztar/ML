# -*- coding: utf-8 -*-
"""

"""
from test_class import dataObject

# Create an instance of training values
train_values = dataObject('train_values.csv')

# Create an instance of training labels
train_labels = dataObject('train_labels.csv')

# Read data
train_values.read_data()
train_labels.read_data()

# Clean the data
train_values.clean()

# Split data into train/test set
train_values.train_test_set(train_labels.df)

# Send training data to select feature importance
train_values.feature_selection(train_values.X_train, train_values.y_train)
