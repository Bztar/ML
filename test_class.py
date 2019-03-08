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
        self.X_train, self.X_test,\
        self.y_train, self.y_test = train_test_split(self.df,
                                                    labels,
                                                    test_size=0.3,
                                                    random_state=0)
    
    def feature_selection(self, X_train, y_train):
        # Use randomforest for feature selection
        self.feat_labels = X_train.columns
        
        # Set up the RF-classifier
        forest = RandomForestClassifier(n_estimators=100,
                                        random_state=0,
                                        n_jobs=-1)
        
        # Fit the clf to the training data
        forest.fit(X_train, y_train)
        
        # Ranking of feature importance
        self.importances = forest.feature_importances_
        self.indices = np.argsort(self.importances)[::-1]

    def show_features(self):
        # Print the results
        for f in range(self.X_train.shape[1]):
            print("%2d. %-*s %f" % (f+1, 50, 
                            self.feat_labels[self.indices[f]],
                            self.importances[self.indices[f]]))
        
        # Plot the results from feature selection in bar plot
        plt.title('Feature Importances')
        plt.bar(range(self.X_train.shape[1]),
                self.importances[self.indices],
                color='lightblue',
                align='center')
        
        plt.xticks(range(self.X_train.shape[1]),
                   self.feat_labels[self.indices], rotation=90)
        
        plt.xlim([-1, self.X_train.shape[1]])
        plt.tight_layout()
        plt.show()



