import pandas as pd
import numpy as np
import data_preprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import f1_score


def KNN_classification(dataset_loader, print_flag=False):
    model_name = 'KNN'
    X_train, y_train, X_test, y_test = dataset_loader
    neigh = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    f1_train = f1_score(y_train, neigh.predict(X_train))
    f1_test = f1_score(y_test, neigh.predict(X_test))
    if print_flag:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The f1 score of train_dataset is {0}".format(f1_train))
        print("The f1 score of test_dataset is {0}".format(f1_test))

    model_evaluation_result = (model_name, f1_train, f1_test)

    return model_evaluation_result


def svm_classification(dataset_loader, print_flag=False):
    model_name = 'SVM'
    X_train, y_train, X_test, y_test = dataset_loader
    clf = svm.SVC(kernel='rbf').fit(X_train, y_train)
    f1_train = f1_score(y_train, clf.predict(X_train))
    f1_test = f1_score(y_test, clf.predict(X_test))
    if print_flag:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The f1 score of train_dataset is {0}".format(f1_train))
        print("The f1 score of test_dataset is {0}".format(f1_test))

    model_evaluation_result = (model_name, f1_train, f1_test)

    return model_evaluation_result


def decision_tree_classification(data_loader,print_flag=False):
    model_name='decision tree'
