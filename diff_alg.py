# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:54:05 2019

@author: mbackstr
"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

"""
KNN -> log loss = 0.38370
"""
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', KNeighborsClassifier())])

grid_params = {
        'clf__n_neighbors': [3,5,11,19],
        'clf__weights': ['uniform', 'distance'], 
        'clf__metric': ['euclidean', 'manhattan']}

gs = GridSearchCV(estimator=pipe_lr,
                  param_grid=grid_params,
                  verbose=1,
                  cv=3,
                  n_jobs=1)

gs = gs.fit(values.X_train, values.y_train)
print(gs.best_score_)
print(gs.best_params_)

from sklearn.metrics import log_loss
y_fake = gs.predict_proba(values.X_test)
print('Log Loss: %.4f' % log_loss(values.y_test, y_fake))

# Choose the column with 1 i.e. proba that patient has heart disease
pred = pd.DataFrame(gs.predict_proba(test.df)[:,1],
                    index=test.df.index,
                    columns=labels.df.columns)

pred.to_csv('submission_KNN.csv')

"""
Support vector -> log loss = 0.46408
"""
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', SVC(random_state=1, probability=True))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
                {'clf__C': param_range, 
                 'clf__gamma': param_range, 
                 'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=1)

gs = gs.fit(values.X_train, values.y_train)
print(gs.best_score_)
print(gs.best_params_)

from sklearn.metrics import log_loss
y_fake = gs.predict_proba(values.X_test)
print(log_loss(values.y_test, y_fake))

# Choose the column with 1 i.e. proba that patient has heart disease
pred = pd.DataFrame(gs.predict_proba(test.df)[:,1],
                    index=test.df.index,
                    columns=labels.df.columns)

pred.to_csv('submission_SVC .csv')

"""
RandomForestClassifier -> log loss = 0.39864
"""

pipe_lr = Pipeline([('clf', RandomForestClassifier(random_state=42))])

param_grid = { 
    'clf__n_estimators': [5000, 10000],
    'clf__max_features': ['auto', 'sqrt', 'log2'],
    'clf__max_depth' : [4,5,6,7,8],
    'clf__criterion' :['gini', 'entropy']
}

gs = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid,
                  cv=5,
                  n_jobs=1)

gs = gs.fit(values.X_train, values.y_train)
print(gs.best_score_)
print(gs.best_params_)

from sklearn.metrics import log_loss
y_fake = gs.predict_proba(values.X_test)
print(log_loss(values.y_test, y_fake))

# Choose the column with 1 i.e. proba that patient has heart disease
pred = pd.DataFrame(gs.predict_proba(test.df)[:,1],
                    index=test.df.index,
                    columns=labels.df.columns)

pred.to_csv('submission_TREE.csv')
