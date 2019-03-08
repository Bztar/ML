# -*- coding: utf-8 -*-
"""

"""
from data_class import dataObject
import pandas as pd

pd.set_option('display.max_columns', 40)

# Create an instance of training values
values = dataObject('train_values.csv')

# Create an instance of training labels
labels = dataObject('train_labels.csv')

# Read data
values.read_data('patient_id')
labels.read_data('patient_id')

# Clean the data
values.clean()

# Split data into train/test set
values.train_test_set(labels.df)

# Send training data to select feature importance
values.feature_selection(values.X_train, values.y_train)

# Show feature importance
values.show_features()
