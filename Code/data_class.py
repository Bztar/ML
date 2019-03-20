#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""
# Mute warnings
import warnings
warnings.filterwarnings("ignore")

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Import cross val and metrics
from sklearn.model_selection import cross_val_score


# Define the class dataObject
class dataObject:
    
    def __init__(self, filename):
        self.filename = filename
        
    def read_data(self, index):
        self.df = pd.read_csv("../Data/"+self.filename, index_col=index)
        return self.df
    
    def clean(self):
        self.df = pd.get_dummies(self.df)

    def train_test_set(self, labels):
        # Separate into training and test sets
        self.X_train, self.X_test,\
        self.y_train, self.y_test = train_test_split(self.df,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=0)
    
    def feature_importance(self, X_train, y_train):
        # Use randomforest for feature selection
        self.feat_labels = X_train.columns
        
        # Set up the RF-classifier
        forest = RandomForestClassifier(n_estimators=100,
                                        random_state=0,
                                        n_jobs=-1)
        
        # Fit the RF-clf to the training data
        forest.fit(X_train, y_train)
        
        # Ranking of feature importance
        self.importances = forest.feature_importances_
        self.indices = np.argsort(self.importances)[::-1]
        
        
        
    def show_features(self):
        # Print the results
        print("List of feature importance")
        for f in range(self.X_train.shape[1]):
            print("%2d. %-*s %f" % (f+1, 40, 
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
        
    def feature_selection():
        """
        TODO: Add vizualisation for optimal # of features
        http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
        """
        pass
        
    def classifier_results(self):
        # Make a list of Classifiers
        classifiers = [LogisticRegression(),
                       KNeighborsClassifier(n_neighbors=3),
                       GaussianNB(),
                       RandomForestClassifier(n_estimators=1000, random_state=8),
                       SVC(probability=True)]
        
        # Cross val scores for the different clf
        for clf in classifiers:
            name = clf.__class__.__name__
            Accuracy = cross_val_score(clf, 
                                       self.X_train, 
                                       self.y_train, 
                                       scoring='accuracy',
                                       cv=5).mean()
            
            LogLoss = cross_val_score(clf, 
                                       self.X_train, 
                                       self.y_train, 
                                       scoring='neg_log_loss',
                                       cv=5).mean()
            
            ROC_AUC = cross_val_score(clf, 
                                       self.X_train, 
                                       self.y_train, 
                                       scoring='roc_auc',
                                       cv=5).mean()
            
            print('='*30)
            print(name)
            
            print('**Results**')
            print('Accuracy: {}'.format(Accuracy))
            print('LogLoss: {}'.format(LogLoss))
            print('ROC_AUC: {}'.format(ROC_AUC))
        
        print('='*30)



