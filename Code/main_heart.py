# -*- coding: utf-8 -*-
"""
https://scikit-learn.org/stable/

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
#values.feature_importance(values.X_train, values.y_train)

# Show feature importance
#values.show_features()


"""
TODO: Move the different steps below to the data_class
"""
# Pipeline
# values.create_pipeline(Scaling, Classifier)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Pipeline for classifier
pipe = Pipeline([('scl', StandardScaler()),
                 ('clf', SVC(random_state=1,
                             probability=True))])

"""
Hyperparameter tuning for the chosen classifier
Returns the best tuned classifier from GridSearch
"""
#clf = values.hypertune
# Parameter range for C and gamma
C_param_range = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
gamma_param_range = [1e-2, 1e-3, 1e-4, 1e-5]

# What types of scoring in gridsearch
scores = ['precision', 'recall','accuracy']

# Three diff kernels: linear, rbf and sigmoid
param_grid = [{'clf__kernel':['linear'],
               'clf__C':C_param_range},
              {'clf__kernel': ['rbf'],
               'clf__C': C_param_range, 
               'clf__gamma': gamma_param_range},
               {'clf__kernel': ['sigmoid'],
                'clf__C': C_param_range,
                'clf__gamma': gamma_param_range}]

for score in scores: 
    print('Hyperparameter tuning for %s' % score)
    print('='*30)
    gs = GridSearchCV(estimator=pipe,
                      param_grid = param_grid,
                      scoring = '%s' % score,
                      cv = 10,
                      n_jobs = 1,
                      return_train_score=True)
    
    gs.fit(values.X_train, values.y_train)

    print("Best parameters set found on development set:")
    print()
    print(gs.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    
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
# 
# =============================================================================
