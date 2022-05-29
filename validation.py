import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import classification_models
import data_preprocess
import regression_models


def classification_cv(dataset, classification_model):
    X, y = dataset
    kf = KFold(n_splits=5, shuffle=True)
    f1_score_train = []
    f1_score_test = []
    for train_index, test_index in tqdm(kf.split(X)):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index].ravel(), y.values[test_index].ravel()
        data_loader = (X_train, y_train, X_test, y_test)
        _, f1_train, f1_test = classification_model(data_loader, print_flag=False)
        f1_score_train.append(f1_train)
        f1_score_test.append(f1_test)

    f1_train_mean = sum(f1_score_train) / len(f1_score_train)
    f1_test_mean = sum(f1_score_test) / len(f1_score_test)
    print('The average f1 score of train dataset is ' + str(f1_train_mean), end=' ')
    print('The average f1 score of test dataset is ' + str(f1_test_mean))
    return f1_train_mean, f1_test_mean


def model_evaluation(model_selection_list: list, csv_path, feature_str: list, non_normalization_feature: list = None,
                     plot_flag=False, task='Regression'):
    data_loader = None
    label = ''
    if task in ['Regression', 'regression']:
        data_loader = data_preprocess.regression_dataloader(csv_path, feature_str, non_normalization_feature)
        label = 'r2'
    elif task in ['classification', 'Classification']:
        data_loader = data_preprocess.classification_dataloader(csv_path, feature_str,
                                                                non_normalization_feature)
        X, y = data_loader
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        data_loader = (X_train.values, y_train.values.ravel(), X_test.values, y_test.values.ravel())
        label = 'f1'
    all_model_evaluation_result = []
    for item in tqdm(model_selection_list):
        single_model_evaluation_result = model_call(model_name=item, dataloader=data_loader)
        all_model_evaluation_result.append(single_model_evaluation_result)

    for item in all_model_evaluation_result:
        print("The model is {0} ".format(item[0]), end="")
        print("train " + label + " score is " + str(item[1]), end=" ")
        print("test " + label + " score is " + str(item[2]))

    if plot_flag:
        compare_model_plot(all_model_evaluation_result, task)


def model_call(model_name: str, dataloader, print_flag=False):
    if model_name in ['KNN', 'knn']:
        return classification_models.KNN_classification(dataloader, print_flag)
    elif model_name in ['svm', 'SVM']:
        return classification_models.svm_classification(dataloader, print_flag)
    elif model_name in ['lda', 'LDA']:
        return classification_models.LDA_classification(dataloader, print_flag)
    elif model_name in ['decision tree']:
        return classification_models.decision_tree_classification(dataloader, print_flag)
    elif model_name in ['logistic regression', 'logistic']:
        return classification_models.logistic_regression_classification(dataloader, print_flag)
    elif model_name in ['mlp classification']:
        return classification_models.mlp_classification(dataloader, print_flag)
    elif model_name == 'ordinary regression':
        return regression_models.ordinary_regression(dataloader, print_flag)
    elif model_name == 'LASSO regression':
        return regression_models.LASSO_regression(dataloader, print_flag)
    elif model_name == 'random forest regressor':
        return regression_models.random_forest_regressor(dataloader, print_flag)
    elif model_name == 'extra trees regressor':
        return regression_models.extra_trees_regressor(dataloader, print_flag)
    elif model_name == 'gradient boosting regressor':
        return regression_models.gradient_boosting_regressor(dataloader, print_flag)
    elif model_name == 'svr':
        return regression_models.svr(dataloader, print_flag)
    elif model_name == 'mlp regressor':
        return regression_models.mlp_regressor(dataloader, print_flag)


def compare_model_plot(model_result_list: list, task='Regression'):
    label = ''
    if task in ['Regression', 'regression']:
        label = 'r2 score'
    elif task in ['Classification', 'classification']:
        label = 'f1 score'

    model_name = []
    train_score = []
    test_score = []
    for item1, item2, item3 in model_result_list:
        model_name.append(item1)
        train_score.append(item2)
        test_score.append(item3)

    bar_width = .35
    x = np.arange(len(model_name))
    plt.figure(figsize=(15, 10))
    plt.bar(x, train_score, bar_width, color='c', align='center', label=('train ' + label))
    plt.bar(x + bar_width, test_score, bar_width, color='b', align='center', label=('test ' + label))
    plt.xlabel("models")
    plt.ylabel(label)
    plt.xticks(x + bar_width / 2, model_name)
    plt.legend()
    plt.savefig('./report_images/compare_model.svg')
    plt.show()


if __name__ == '__main__':
    feature_str = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'feature 1']
    non_normalization_feature = ['cbwd', 'feature 1']
    dataloader = data_preprocess.classification_dataloader('new_feature.csv', selected_feature=feature_str,
                                                           non_normalization_feature=non_normalization_feature)
    classification_cv(dataloader, classification_models.svm_classification)
