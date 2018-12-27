#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'total_payments',
                 'expenses', 'long_term_incentive'] # You will need to use more features

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

my_dataset.pop('TOTAL')

from poi_utils import include_perc_poi_messages

features_list.append('perc_poi_messages')
include_perc_poi_messages(my_dataset, 'perc_poi_messages',
                          ['from_this_person_to_poi', 'from_poi_to_this_person', 'shared_receipt_with_poi'],
                          ['to_messages', 'from_messages', 'shared_receipt_with_poi'])

features_list.append('perc_this_person_to_poi')
include_perc_poi_messages(my_dataset, 'perc_this_person_to_poi', ['from_this_person_to_poi'], ['from_messages'])

features_list.append('perc_poi_to_this_person')
include_perc_poi_messages(my_dataset, 'perc_poi_to_this_person', ['from_poi_to_this_person'], ['to_messages'])

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from poi_utils import showOnPlot
from poi_utils import showNaNs
#showNaNs(features, features_list[1:])
#showOnPlot(features, labels, 0, 8, features_list)

from sklearn.preprocessing import MinMaxScaler
features = MinMaxScaler().fit_transform(features)

#showOnPlot(features, labels, 4, 5, features_list)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree

pca = PCA(n_components=8)
features = pca.fit_transform(features)

#clf = GaussianNB()
clf = svm.SVC(gamma="auto", C=8000.0, kernel='rbf')
#clf = tree.DecisionTreeClassifier()

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

from sklearn.model_selection import GridSearchCV
from sklearn import svm
from time import time

print "Fitting the classifier to the training set"
t0 = time()

#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100, 1000, 3000, 5000, 8000, 9000, 10000, 11000]}
#clf = GridSearchCV(svm.SVC(gamma="auto"), parameters, cv=5, iid=False, scoring='f1')
clf.fit(features_train, labels_train)

#clf.fit(features_train, labels_train)

print "done in %0.3fs" % (time() - t0)
#print "Best estimator found by grid search:"
#print clf.best_params_

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print "Accuracy Train", round(clf.score(features_train, labels_train), 2)
print "Accuracy Test", round(clf.score(features_test, labels_test), 2)

prediction_test = clf.predict(features_test)
print "Precision Score", round(precision_score(labels_test, prediction_test, average='binary'), 2)
print "Recall Score", round(recall_score(labels_test, prediction_test, average='binary'), 2)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)