import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm

import dataPreprocess

# 输入数据 返回模型预测值

'''
param 
    dataset_loader_np reference to function: csv_to_dataset_np
return 
    type tuple(model_name:str,train_score:str,test_score_str)
'''


# 目前默认只有r2的评价标准


def ordinary_regression(dataset_loader_np, flag_print=False):
    model_name = 'ordinary regression'
    X_train, y_train, X_test, y_test = dataset_loader_np
    ordinary_regression_model = LinearRegression().fit(X_train, y_train)
    train_score = r2_score(y_train, ordinary_regression_model.predict(X_train))
    test_score = r2_score(y_test, ordinary_regression_model.predict(X_test))

    if flag_print:
        # 打印当前的模型信息
        print("The model is {0}".format('ordinary regression model'))
        print("The R2 score of train_dataset is {0}".format(train_score))
        print("The R2 score of test_dataset is {0}".format(test_score))

    model_evaluation_result = (model_name, train_score, test_score)

    return model_evaluation_result


def LASSO_regression(dataset_loader_np, flag_print=False):
    model_name = 'LASSO regression'

    X_train, y_train, X_test, y_test = dataset_loader_np

    LASSO_regression_model = Lasso().fit(X_train, y_train)
    train_score = r2_score(y_train, LASSO_regression_model.predict(X_train))
    test_score = r2_score(y_test, LASSO_regression_model.predict(X_test))

    if flag_print:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The R2 score of train_dataset is {0}".format(train_score))
        print("The R2 score of test_dataset is {0}".format(test_score))

    model_evaluation_result = (model_name, train_score, test_score)

    return model_evaluation_result


def ridge_regression(dataset_loader_np, flag_print=False):
    model_name = 'ridge regression'
    selected_variable_lasso = []

    X_train, y_train, X_test, y_test = dataset_loader_np

    ridge_regression_model = Ridge().fit(X_train, y_train)
    train_score = r2_score(y_train, ridge_regression_model.predict(X_train))
    test_score = r2_score(y_test, ridge_regression_model.predict(X_test))

    if flag_print:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The R2 score of train_dataset is {0}".format(train_score))
        print("The R2 score of test_dataset is {0}".format(test_score))

    model_evaluation_result = (model_name, train_score, test_score)

    return model_evaluation_result


def random_forest_regressor(dataset_loader_np, flag_print=False):
    model_name = 'random forest regressor'
    X_train, y_train, X_test, y_test = dataset_loader_np

    random_forest_regressor_model = RandomForestRegressor().fit(X_train, y_train)
    train_score = r2_score(y_train, random_forest_regressor_model.predict(X_train))
    test_score = r2_score(y_test, random_forest_regressor_model.predict(X_test))

    if flag_print:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The R2 score of train_dataset is {0}".format(train_score))
        print("The R2 score of test_dataset is {0}".format(test_score))

    model_evaluation_result = (model_name, train_score, test_score)

    return model_evaluation_result


def extra_trees_regressor(dataset_loader_np, flag_print=False):
    model_name = 'extra trees regressor'
    X_train, y_train, X_test, y_test = dataset_loader_np

    extra_trees_regressor_model = ExtraTreesRegressor().fit(X_train, y_train)
    train_score = r2_score(y_train, extra_trees_regressor_model.predict(X_train))
    test_score = r2_score(y_test, extra_trees_regressor_model.predict(X_test))

    if flag_print:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The R2 score of train_dataset is {0}".format(train_score))
        print("The R2 score of test_dataset is {0}".format(test_score))

    model_evaluation_result = (model_name, train_score, test_score)

    return model_evaluation_result


def gradient_boosting_regressor(dataset_loader_np, flag_print=False):
    X_train, y_train, X_test, y_test = dataset_loader_np

    model_name = 'gradient boosting regressor'
    gradient_boosting_regressor_model = GradientBoostingRegressor().fit(X_train, y_train)
    train_score = r2_score(y_train, gradient_boosting_regressor_model.predict(X_train))
    test_score = r2_score(y_test, gradient_boosting_regressor_model.predict(X_test))

    if flag_print:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The R2 score of train_dataset is {0}".format(train_score))
        print("The R2 score of test_dataset is {0}".format(test_score))

    model_evaluation_result = (model_name, train_score, test_score)

    return model_evaluation_result


def svr(dataset_loader_np, flag_print=False, param_c=1.2):
    model_name = 'svr'
    X_train, y_train, X_test, y_test = dataset_loader_np

    svr_model = svm.SVR(C=param_c).fit(X_train, y_train)
    train_score = r2_score(y_train, svr_model.predict(X_train))
    test_score = r2_score(y_test, svr_model.predict(X_test))

    if flag_print:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The R2 score of train_dataset is {0}".format(train_score))
        print("The R2 score of test_dataset is {0}".format(test_score))

    model_evaluation_result = (model_name, train_score, test_score)

    return model_evaluation_result


'''
model_selection_list str for model_name
'''


def model_evaluation(model_selection_list: list, csv_path, feature_str: list, non_normalization_feature: list = None,
                     print_flag=False):
    dataset_loader_np = dataPreprocess.regression_dataloader(csv_path, feature_str, non_normalization_feature)
    all_model_evaluation_result = []
    for item in tqdm(model_selection_list):
        single_model_evaluation_result = model_call(item, dataset_loader_np, print_flag)
        all_model_evaluation_result.append(single_model_evaluation_result)

    # print part
    for item in all_model_evaluation_result:
        print("The model is {0} ".format(item[0]), end="")
        print("train_r2 is {0} ".format(item[1]), end="")
        print("test_r2 is {0}".format(item[2]))

    compare_model(all_model_evaluation_result)

    return all_model_evaluation_result


def model_call(model_name: str, dataset_loader_np, print_flag=False):
    if model_name == 'ordinary regression':
        return ordinary_regression(dataset_loader_np, print_flag)
    elif model_name == 'LASSO regression':
        return LASSO_regression(dataset_loader_np, print_flag)
    elif model_name == 'random forest regressor':
        return random_forest_regressor(dataset_loader_np, print_flag)
    elif model_name == 'extra trees regressor':
        return extra_trees_regressor(dataset_loader_np, print_flag)
    elif model_name == 'gradient boosting regressor':
        return gradient_boosting_regressor(dataset_loader_np, print_flag)
    elif model_name == 'svr':
        return svr(dataset_loader_np, print_flag)


def compare_model(model_result_list: list):
    model_name = []
    train_r2 = []
    test_r2 = []
    for item1, item2, item3 in model_result_list:
        model_name.append(item1)
        train_r2.append(item2)
        test_r2.append(item3)

    bar_width = .35
    x = np.arange(len(model_name))
    plt.figure(figsize=(15, 10))
    plt.bar(x, train_r2, bar_width, color='c', align='center', label='train r2')
    plt.bar(x + bar_width, test_r2, bar_width, color='b', align='center', label='test r2')
    plt.xlabel("models")
    plt.ylabel("r2")
    plt.xticks(x + bar_width / 2, model_name)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    '''
    Models include ordinary regression, LASSO regression, Random Forest regressor, Extra trees regressor, \
    Gradient Boosting regressor, SVR
    '''

    model_selection_list = ['ordinary regression', 'LASSO regression', 'random forest regressor',
                            'extra trees regressor', 'gradient boosting regressor']
    csv_path = './new_feature.csv'
    feature_str = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir', 'feature 1']
    non_normalization_feature = ['cbwd', 'feature 1']
    model_evaluation(model_selection_list, csv_path, feature_str, non_normalization_feature)
