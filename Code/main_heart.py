# -*- coding: utf-8 -*-
"""
https://scikit-learn.org/stable/
https://github.com/sujoyde89/driven-data-kernels
"""
from data_class import dataObject
import pandas as pd
pd.set_option('display.max_columns', 40)

# Create an instance of training values
values = dataObject('train_values.csv')

# Create an instance of training labels
labels = dataObject('train_labels.csv')

# Create an instance of testdata
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

# Which classifier shows best results (naive)
#values.classifier_results()

# Send training data to select feature importance
#values.feature_selection(values.X_train, values.y_train)

# Show feature importance
#values.show_features()

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scl', StandardScaler()),
                 ('clf', LogisticRegression())])

# Create the random grid
param_grid = {}

gs = GridSearchCV(estimator=pipe,
                  param_grid = param_grid,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = 1)

gs = gs.fit(values.X_train, values.y_train)
print(gs.best_score_)

clf = gs.best_estimator_
clf.fit(values.X_train, values.y_train)
print('Test accuracy: %.3f' % clf.score(values.X_test, values.y_test))

# =============================================================================
# # Predict on test values, first column = 0, second = 1
# pred = pd.DataFrame(clf.predict_proba(test.df))
# pred.columns = ['No','heart_disease_present']
# 
# res = pred[['heart_disease_present']]
# res.index = test.df.index
# res.to_csv('result.csv')
# =============================================================================


