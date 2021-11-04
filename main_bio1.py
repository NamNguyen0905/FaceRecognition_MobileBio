''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import get_images
import get_landmarks
import performance_plots

from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
import pandas as pd

''' Import classifiers '''
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svm
from sklearn.naive_bayes import GaussianNB


''' Load the data and their labels '''
image_directory = '../Caltech Faces Dataset'
X, y = get_images.get_images(image_directory)

''' Get distances between face landmarks in the images '''
X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 5, False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

''' Matching and Decision - Classifer 1 '''
clf = ORC(knn())
clf.fit(X_train, y_train)
matching_scores_knn = clf.predict_proba(X_val)

''' Matching and Decision - Classifer 2 '''
clf = ORC(GaussianNB())
clf.fit(X_train, y_train)
matching_scores_NB = clf.predict_proba(X_val)

''' Fusing score kNN and NB of validated data and calculate the threshold'''
matching_scores_knn_NB = (matching_scores_knn + matching_scores_NB) / 2.0

# Tuning the sytem
gen_scores = []
imp_scores = []
classes = clf.classes_
matching_scores_knn_NB = pd.DataFrame(matching_scores_knn_NB, columns=classes)

for i in range(len(y_val)):    
    scores = matching_scores_knn_NB.loc[i]
    mask = scores.index.isin([y_val[i]])
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])
    
threshold_knn_NB = performance_plots.performance(gen_scores, imp_scores, 'kNN_NB_decision_fusion', 100)

'''---'''

# Testing the system - getting a decision for kNN classsifier
matching_scores_knn = clf.predict_proba(X_test)

# Testing the system - getting a decision for NB classsifier
matching_score_NB = clf.predict_proba(X_test)

''' Fusing score kNN and NB of testing data'''
matching_scores_knn_NB = (matching_scores_knn + matching_scores_NB) / 2.0
matching_scores_knn_NB = pd.DataFrame(matching_scores_knn_NB, columns=classes)

gen_scores_knn_NB = []
imp_scores_knn_NB = []
for i in range(len(y_test)):    
    scores = matching_scores_knn_NB.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores_knn_NB.extend(scores[mask])
    imp_scores_knn_NB.extend(scores[~mask])

'''
-----------------------------------------------------------------
'''

''' Matching and Decision - Classifer 3 '''
clf = ORC(svm(probability=True))
clf.fit(X_train, y_train)
matching_scores_svm = clf.predict_proba(X_val)

# Tuning the sytem
gen_scores = []
imp_scores = []
classes = clf.classes_
matching_scores_svm = pd.DataFrame(matching_scores_svm, columns=classes)

for i in range(len(y_val)):    
    scores = matching_scores_svm.loc[i]
    mask = scores.index.isin([y_val[i]])
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])
    
threshold_svm = performance_plots.performance(gen_scores, imp_scores, 'SVM_decision_fusion', 100)

# Testing the system - getting a decision
matching_scores_svm = clf.predict_proba(X_test)
matching_scores_svm = pd.DataFrame(matching_scores_svm, columns=classes)

gen_scores_svm = []
imp_scores_svm = []
for i in range(len(y_test)):    
    scores = matching_scores_svm.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores_svm.extend(scores[mask])
    imp_scores_svm.extend(scores[~mask])
    
'''
Fuse decisions
'''
correct_authentications = 0
for i in range(len(gen_scores_knn_NB)):
    decision_knn_NB = False
    decision_svm = False
    if gen_scores_knn_NB[i] >= threshold_knn_NB:
        decision_knn_NB = True
        if gen_scores_svm[i] >= threshold_svm:
            decision_svm = True
            if decision_knn_NB and decision_svm:
                correct_authentications += 1
                
for i in range(len(imp_scores_knn_NB)):
    decision_knn_NB = False
    decision_svm_NB = False
    if imp_scores_knn_NB[i] < threshold_knn_NB:
        decision_knn_NB = True
        if imp_scores_svm[i] < threshold_svm:
            decision_svm = True
            if decision_knn_NB and decision_svm:
                correct_authentications += 1

all_authentications = len(gen_scores_knn_NB) + len(imp_scores_knn_NB)
accuracy = correct_authentications / all_authentications
