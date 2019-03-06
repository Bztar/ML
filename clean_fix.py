# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:22:52 2019

@author: mbackstr
"""
from sklearn.model_selection import train_test_split

def clean_fix(train_values, train_labels, test_values):
    
    # Map the data in thal to integers
    thal_map = {'normal': 0, 
                'reversible_defect': 1,
                'fixed_defect': 2}

    train_values.thal = train_values.thal.map(thal_map)
    test_values.thal = test_values.thal.map(thal_map)
    
    # Separate into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(train_values,
                                                        train_labels,
                                                        test_size=0.3,
                                                        random_state=0)
    
    return X_train, X_test, y_train, y_test