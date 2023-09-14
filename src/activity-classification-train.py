# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle

import labels


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
# REPLACE WITH USER PATH TO FOLDER
data_file = f'{os.path.dirname(os.path.realpath(__file__))}/data/all_labeled_data.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,2], data[i,3], data[i,4]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:2],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

data = np.nan_to_num(data)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

# sampling rate should be about 100 Hz (sensor logger app); you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,1] - data[0,1])
sampling_rate = n_samples / time_elapsed_seconds

print("Sampling Rate: " + str(sampling_rate))

# list the class labels that you collected data for in the order of label_index (defined in labels.py)
class_names = labels.activity_labels

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []
feature_names = []
for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,2:-1]
    # print("window = ")
    # print(window)
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
    
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


# split data into train and test datasets using 10-fold cross validation
kf = KFold(n_splits=10, random_state=None, shuffle=True)
    
"""
Iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""
n_fold = 1
acc_vals = []
prec_vals = []
rec_vals = []

for train_i, test_i in kf.split(X, Y):
    # Train dataset
    x_train, y_train = X[train_i], Y[train_i]
    # Test dataset
    x_test, y_test = X[test_i], Y[test_i]
    
    # Decision tree classifier fit to training dataset
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    tree.fit(x_train,y_train)
    
    # Y = class labels, Y_test = ground truth vals, prediction and confusion matrix
    y_pred = tree.predict(x_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    acc_val = accuracy_score(y_test, y_pred)
    prec_val = precision_score(y_test, y_pred, average='macro')
    rec_val = recall_score(y_test, y_pred, average='macro')
    
    # Print and append values, inc fold counter
    print("Current Fold: ", n_fold)
    print("Accuracy: ", acc_val)
    print("Precision: ", prec_val)
    print("Recall: ", rec_val)
    print("\n")
    
    n_fold += 1
    acc_vals.append(acc_val)
    prec_vals.append(prec_val)
    rec_vals.append(rec_val)

# Calculate and print the average accuracy, precision and recall values over all 10 folds
acc_avg = np.mean(acc_vals)
prec_avg = np.mean(prec_vals)
rec_avg = np.mean(rec_vals)

print("Average Accuracy:", acc_avg)
print("Average Precision:", prec_avg)
print("Average Recall:", rec_avg)
print("\n")

# Train the decision tree classifier on entire dataset
tree = DecisionTreeClassifier()
tree.fit(X,Y)

# Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
export_graphviz(tree, out_file='tree.dot', feature_names = feature_names)

# Save the classifier to disk - replace 'tree' with your decision tree and run the below line
print("saving classifier model...")
with open('classifier.pickle', 'wb') as f:
    pickle.dump(tree, f)