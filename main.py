# -*- coding: utf-8 -*-
"""

"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import data
train_values = pd.read_csv('train_values.csv', index_col='patient_id')
train_labels = pd.read_csv('train_labels.csv', index_col='patient_id')
test_values = pd.read_csv('test_values.csv', index_col='patient_id')


"""
Set up a structure for cleaning and feature selection
    - Take training and test data
    - return data with selected features
"""
from clean_fix import clean_fix
X_train, X_test, y_train, y_test = clean_fix(train_values, 
                                             train_labels, 
                                             test_values)


"""
Set up a pipeline
    - In: classifier
    - Return pipeline
"""
#pipe_lr = Pipeline([('scl', StandardScaler()), 
                    #('clf', LogisticRegression())]) 

#pipe_lr.fit(X_train, y_train)
#print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))


"""
Set up structure for gridSearchCV
    - Takes pipeline
"""