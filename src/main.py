''' Imports '''
import warnings
from sklearn.naive_bayes import GaussianNB # Classifier
from sklearn.svm import SVC as svm # Classifier
from sklearn.neighbors import KNeighborsClassifier as knn # Classifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier as ORC
import performance_plots
import get_landmarks
import get_images
import numpy as np


def warn(*args, **kwargs):
    pass

def split_dataset(X, y, imgs_quality, qual, hqTest):
    if not qual:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
    else:
        X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
            
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        if hqTest:
            for X, y, img_quality in zip(X_temp, y_temp, imgs_quality):
                if img_quality[0] < 1080 or img_quality[0] < 1080:
                    X_train.append(X)
                    y_train.append(y)
                else:
                    X_test.append(X)
                    y_test.append(y)
        else:
            for X, y, img_quality in zip(X_temp, y_temp, imgs_quality):
                if img_quality[0] >= 1080 or img_quality[0] >= 1080:
                    X_train.append(X)
                    y_train.append(y)
                else:
                    X_test.append(X)
                    y_test.append(y)
        
    return X_train, X_test, X_val, y_train, y_test, y_val

def train_kNN_NB(X_train, y_train, X_val, y_val, X_test, y_test):
    ''' Matching and Decision - Classifer 1 '''
    clf = ORC(knn())
    clf.fit(X_train, y_train)
    matching_scores_knn = clf.predict_proba(X_val)
    
    ''' Matching and Decision - Classifer 2 '''
    clf = ORC(GaussianNB())
    clf.fit(X_train, y_train)
    matching_scores_NB = clf.predict_proba(X_val)
    
    ''' Fusing score kNN and NB of validated data and calculate the threshold'''
    matching_scores_knn_NB = np.mean( np.array([matching_scores_knn, matching_scores_NB]), axis=0 )
    
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
    matching_scores_NB = clf.predict_proba(X_test)
    
    ''' Fusing score kNN and NB of testing data'''
    matching_scores_knn_NB = np.mean( np.array([matching_scores_knn, matching_scores_NB]), axis=0 )
    matching_scores_knn_NB = pd.DataFrame(matching_scores_knn_NB, columns=classes)
    
    gen_scores_knn_NB = []
    imp_scores_knn_NB = []
    for i in range(len(y_test)):
        scores = matching_scores_knn_NB.loc[i]
        mask = scores.index.isin([y_test[i]])
        gen_scores_knn_NB.extend(scores[mask])
        imp_scores_knn_NB.extend(scores[~mask])
        
    return threshold_knn_NB, gen_scores_knn_NB, imp_scores_knn_NB

def train_SVM(X_train, y_train, X_val, y_val, X_test, y_test):
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
    
    threshold_svm = performance_plots.performance(
        gen_scores, imp_scores, 'SVM_decision_fusion', 100)
    
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
        
    return threshold_svm, gen_scores_svm, imp_scores_svm

def fuse_decision(gen_scores_knn_NB, imp_scores_knn_NB, gen_scores_svm, imp_scores_svm, threshold_knn_NB, threshold_svm):
    ''' Fuse decisions '''
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
        decision_svm = False
        if imp_scores_knn_NB[i] < threshold_knn_NB:
            decision_knn_NB = True
            if imp_scores_svm[i] < threshold_svm:
                decision_svm = True
                if decision_knn_NB and decision_svm:
                    correct_authentications += 1
    
    all_authentications = len(gen_scores_knn_NB) + len(imp_scores_knn_NB)
    accuracy = correct_authentications / all_authentications
    
    return accuracy

def feature_performance(X, y, imgs_quality):
    ''' Split dataset '''
    # Base spliting
    X_train, X_test, X_val, y_train, y_test, y_val = split_dataset(X, y, imgs_quality, False, False)
    threshold_knn_NB, gen_scores_knn_NB, imp_scores_knn_NB = train_kNN_NB(X_train, y_train, X_val, y_val, X_test, y_test)
    threshold_svm, gen_scores_svm, imp_scores_svm = train_SVM(X_train, y_train, X_val, y_val, X_test, y_test)
    accuracy_base = fuse_decision(gen_scores_knn_NB, imp_scores_knn_NB, gen_scores_svm, imp_scores_svm, threshold_knn_NB, threshold_svm)
    
    
    # High-quality TEST / Low-quality TRAIN
    X_train, X_test, X_val, y_train, y_test, y_val = split_dataset(X, y, imgs_quality, True, True)
    threshold_knn_NB, gen_scores_knn_NB, imp_scores_knn_NB = train_kNN_NB(X_train, y_train, X_val, y_val, X_test, y_test)
    threshold_svm, gen_scores_svm, imp_scores_svm = train_SVM(X_train, y_train, X_val, y_val, X_test, y_test)
    accuracy_HQ = fuse_decision(gen_scores_knn_NB, imp_scores_knn_NB, gen_scores_svm, imp_scores_svm, threshold_knn_NB, threshold_svm)
    
    
    # Low-quality TEST / High-quality TRAIN
    X_train, X_test, X_val, y_train, y_test, y_val = split_dataset(X, y, imgs_quality, True, False)
    threshold_knn_NB, gen_scores_knn_NB, imp_scores_knn_NB = train_kNN_NB(X_train, y_train, X_val, y_val, X_test, y_test)
    threshold_svm, gen_scores_svm, imp_scores_svm = train_SVM(X_train, y_train, X_val, y_val, X_test, y_test)
    accuracy_LQ = fuse_decision(gen_scores_knn_NB, imp_scores_knn_NB, gen_scores_svm, imp_scores_svm, threshold_knn_NB, threshold_svm)
    
    return accuracy_base, accuracy_HQ, accuracy_LQ

def main(): 
    warnings.warn = warn
    
    
    ''' Load the data and their labels '''
    image_directory = '../Project 1 Database'
    face_imgs, labels, imgs_quality = get_images.get_images(image_directory)
    
    ''' Get distances between face landmarks in the images '''
    without_hair, with_hair, wear_mask, labels = get_landmarks.get_landmarks(face_imgs, labels, 'landmarks/', 68, False)
    

    ''' Without facial hair feature '''
    accuracy_base, accuracy_HQ, accuracy_LQ =  feature_performance(without_hair, labels, imgs_quality)
    print("Accuracy for not separating image quality: %.2f" % accuracy_base)
    print("Accuracy for getting high-quality image as test data: %.2f" % accuracy_HQ)
    print("Accuracy for getting low-quality image as test data: %.2f" % accuracy_LQ)
    
    
    ''' With facial hair feature '''
    accuracy_base, accuracy_HQ, accuracy_LQ =  feature_performance(with_hair, labels, imgs_quality)
    print("Accuracy for not separating image quality: %.2f" % accuracy_base)
    print("Accuracy for getting high-quality image as test data: %.2f" % accuracy_HQ)
    print("Accuracy for getting low-quality image as test data: %.2f" % accuracy_LQ)
    
    
    ''' Wearing mask feature '''
    accuracy_base, accuracy_HQ, accuracy_LQ =  feature_performance(wear_mask, labels, imgs_quality)
    print("Accuracy for not separating image quality: %.2f" % accuracy_base)
    print("Accuracy for getting high-quality image as test data: %.2f" % accuracy_HQ)
    print("Accuracy for getting low-quality image as test data: %.2f" % accuracy_LQ)
    
    
main()