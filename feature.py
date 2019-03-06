# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:42:06 2019

@author: mbackstr
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def feature_selection(X_train, y_train, labels):
    
    forest = RandomForestClassifier(n_estimators=10000,
                                    random_state=0,
                                    n_jobs=-1)
    
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for f in range(X_train.shape[1]):
        print('%2d. %-*s %f' % (f+1, 30,
                                feat_labels[indices[f]],
                                importances[indices[f]]))
    
    
    