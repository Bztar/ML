#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Better design
"""
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Define the class dataObject
class dataObject:
    
    def __init__(self, filename):
        self.filename = filename
        
    def read_data(self):
        self.df = pd.read_csv(self.filename, index_col='patient_id')
        return self.df
    
    def clean(self):
        # Map the data in thal column to integers
        thal_map = {'normal': 0, 
                    'reversible_defect': 1,
                    'fixed_defect': 2}
    
        self.df.thal = self.df.thal.map(thal_map)
    
    def train_test_set(self, labels):
        # Separate into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.df,
                                                            labels,
                                                            test_size=0.3,
                                                            random_state=0)
    
        return X_train, X_test, y_train, y_test
    
    def feature_selection(self, X_train, y_train):
        # Use randomforest for feature selection
        feat_labels = X_train.columns
        
        # Set up the RF-classifier
        forest = RandomForestClassifier(n_estimators=100,
                                        random_state=0,
                                        n_jobs=-1)
        
        # Fit the clf to the training data
        forest.fit(X_train, y_train)
        
        # Ranking of feature importance
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Print the results
        for f in range(X_train.shape[1]):
            print("%2d. %-*s %f" % (f+1, 50, 
                            feat_labels[indices[f]],
                            importances[indices[f]]))
        
        # Plot the results from feature selection in bar plot
        plt.title('Feature Importances')
        plt.bar(range(X_train.shape[1]),
                importances[indices],
                color='lightblue',
                align='center')
        
        plt.xticks(range(X_train.shape[1]),
                   feat_labels[indices], rotation=90)
        
        plt.xlim([-1, X_train.shape[1]])
        plt.tight_layout()
        plt.show()


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
X_train, X_test, y_train, y_test = train_values.train_test_set(train_labels.df)

# Send training data to select feature importance
train_values.feature_selection(X_train, y_train)

