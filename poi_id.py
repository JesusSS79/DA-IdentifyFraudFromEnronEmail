#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import poi_utils

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'exercised_stock_options',
                 'total_stock_value',
                 'bonus',
                 'salary',
                 'deferred_income',
                 'long_term_incentive',
                 'restricted_stock',
                 'total_payments',
                 'shared_receipt_with_poi',
                 'loan_advances',
                 'expenses',
 #                'from_poi_to_this_person',
                 'other',
 #                'from_this_person_to_poi',
                 'director_fees',
 #                'to_messages',
                 'deferral_payments',
 #                'from_messages',
                 'restricted_stock_deferred'
                ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "Population: ", len(data_dict)
print "Property Number: ", len(data_dict.values()[0])
print "Property Names:", data_dict.values()[0].keys()

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#poi_utils.show_person_names(my_dataset)

my_dataset.pop('TOTAL')
my_dataset.pop('THE TRAVEL AGENCY IN THE PARK')

poi_utils.include_div_poi(my_dataset, features_list, 'perc_poi_messages',
                          ['from_this_person_to_poi', 'from_poi_to_this_person', 'shared_receipt_with_poi'],
                          ['to_messages', 'from_messages', 'shared_receipt_with_poi'])
poi_utils.include_div_poi(my_dataset, features_list, 'perc_this_person_to_poi', ['from_this_person_to_poi'], ['from_messages'])
poi_utils.include_div_poi(my_dataset, features_list, 'perc_poi_to_this_person', ['from_poi_to_this_person'], ['to_messages'])
poi_utils.include_div_poi(my_dataset, features_list, 'bonus_salary', ['bonus'], ['salary'])
poi_utils.include_div_poi(my_dataset, features_list, 'incentive_salary', ['long_term_incentive'], ['salary'])
poi_utils.include_div_poi(my_dataset, features_list, 'stock_payment', ['total_stock_value'], ['total_payments'])
poi_utils.include_add_poi(my_dataset, features_list, 'total', ['total_payments', 'total_stock_value'])
poi_utils.include_div_poi(my_dataset, features_list, 'total_salary', ['total'], ['salary'])

#poi_utils.show_person_properties(my_dataset, 'restricted_stock_deferred', 15000000, 'upper')
#Wrong data on 'exercised_stock_options' and 'restricted_stock' (data exchange)
my_dataset.pop('BHATNAGAR SANJAY')

#poi_utils.show_person_properties(my_dataset, 'restricted_stock_deferred', 0, 'upper')
#'restricted_stock_deferred' must be negative
my_dataset.pop('BELFER ROBERT')

#poi_utils.show_person_properties(my_dataset, 'payment_stock', 1000, 'upper')
#Total Payment 475. No other payment data and only stock value.
my_dataset.pop('HAUG DAVID L')


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

poi_utils.show_nans(features, features_list[1:])
poi_utils.show_poi_balance(labels)
#poi_utils.show_on_plot(features, labels, 3, 22, features_list)

#for i in range(len(features_list)):
#   poi_utils.show_on_plot(features, labels, 3, i, features_list)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

print "Fitting the classifier to the training set"
t0 = time()

# GaussianNB
pipe_nbc = Pipeline([
        ('scaler', MinMaxScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA()),
        ('classifier', GaussianNB())
    ])

param_grid_nbc = {
    'scaler':                [None, MinMaxScaler()],
    'selector__k':           [10, 11, 12, 14, 16, 'all'],
    'reducer__n_components': [5, 6, 7, 8],
}

pipe = pipe_nbc
param_grid = param_grid_nbc

# DecisionTreeClassifier
pipe_dtc = Pipeline([
        ('scaler', None),
        ('selector', SelectKBest()),
        ('reducer', PCA()),
        ('classifier', DecisionTreeClassifier())
    ])

param_grid_dtc = {
    'scaler':                        [None, MinMaxScaler()],
    'selector__k':                   [10, 11, 12, 14, 16, 'all'],
    'reducer__n_components':         [5, 6, 7, 8],
    'classifier__criterion':         ['gini', 'entropy'],
    'classifier__splitter':          ['best', 'random'],
    'classifier__min_samples_split': [2, 3, 4, 5],
    'classifier__class_weight':      ['balanced', None],
    'classifier__min_samples_leaf':  [1, 2, 3, 4],
    'classifier__max_depth':         [None, 5, 10, 20],
}

pipe = pipe_dtc
param_grid = param_grid_dtc


# KNeighborsClassifier
pipe_knc = Pipeline([
        ('scaler', MinMaxScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA()),
        ('classifier', KNeighborsClassifier())
    ])

param_grid_knc = {
    'scaler':                  [None, MinMaxScaler()],
    'selector__k':             [10, 11, 12, 14, 15, 18, 'all'],
    'reducer__n_components':   [5, 6, 7, 8, 10],
    'classifier__n_neighbors': [2, 3, 4, 5, 6],
    'classifier__weights':     ['uniform', 'distance'],
    'classifier__algorithm':   ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'classifier__leaf_size':   [20, 30, 40],
    'classifier__p':           [1, 2, 3],
}

pipe = pipe_knc
param_grid = param_grid_knc


# GridSearchCV
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss, verbose=1, n_jobs=2)
grid = grid.fit(features_train, labels_train)

print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
print grid.best_params_

#Validation

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print "Accuracy Train", round(grid.score(features_train, labels_train), 2)
print "Accuracy Test", round(grid.score(features_test, labels_test), 2)

prediction_test = grid.predict(features_test)
print "Precision Score", round(precision_score(labels_test, prediction_test, average='binary'), 2)
print "Recall Score", round(recall_score(labels_test, prediction_test, average='binary'), 2)

dump_classifier_and_data(grid.best_estimator_, my_dataset, features_list)