import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import svm
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
    model_name = 'ordinary_regression'
    X_train, y_train, X_test, y_test = dataset_loader_np[0], dataset_loader_np[1], dataset_loader_np[2], \
                                       dataset_loader_np[3]
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
    model_name = 'LASSO_regression'
    selected_variable_lasso = []

    X_train, y_train, X_test, y_test = dataset_loader_np[0], dataset_loader_np[1], dataset_loader_np[2], \
                                       dataset_loader_np[3]

    LASSO_regression_model = Lasso().fit(X_train, y_train)
    train_score = r2_score(y_train, LASSO_regression_model.predict(X_train))
    test_score = r2_score(y_test, LASSO_regression_model.predict(X_test))

    if flag_print:
        # 打印当前的模型信息
        print("The model is {0}".format('LASSO regression model'))
        print("The R2 score of train_dataset is {0}".format(train_score))
        print("The R2 score of test_dataset is {0}".format(test_score))

    model_evaluation_result = (model_name, train_score, test_score)

    return model_evaluation_result


def randomforest_regressor(dataset_loader_np, flag_print=False):
    model_name = 'randomforest_regressor'
    X_train, y_train, X_test, y_test = dataset_loader_np[0], dataset_loader_np[1], dataset_loader_np[2], \
                                       dataset_loader_np[3]

    randomforest_regressor_model = RandomForestRegressor()
    randomforest_regressor_model.fit(X_train, y_train)
    train_score = r2_score(y_train, randomforest_regressor_model.predict(X_train))
    test_score = r2_score(y_test, randomforest_regressor_model.predict(X_test))

    if flag_print:
        # 打印当前的模型信息
        print("The model is {0}".format('randomforest regressor model'))
        print("The R2 score of train_dataset is {0}".format(train_score))
        print("The R2 score of test_dataset is {0}".format(test_score))

    model_evaluation_result = (model_name, train_score, test_score)

    return model_evaluation_result


def extratrees_regressor(dataset_loader_np, flag_print=False):
    model_name = 'extratrees_regressor'
    X_train, y_train, X_test, y_test = dataset_loader_np[0], dataset_loader_np[1], dataset_loader_np[2], \
                                       dataset_loader_np[3]

    extratrees_regressor_model = ExtraTreesRegressor()
    extratrees_regressor_model.fit(X_train, y_train)
    train_score = r2_score(y_train, extratrees_regressor_model.predict(X_train))
    test_score = r2_score(y_test, extratrees_regressor_model.predict(X_test))

    if flag_print:
        # 打印当前的模型信息
        print("The model is {0}".format('extratrees regressor model'))
        print("The R2 score of train_dataset is {0}".format(train_score))
        print("The R2 score of test_dataset is {0}".format(test_score))

    model_evaluation_result = (model_name, train_score, test_score)

    return model_evaluation_result


def gradient_boosting_regressor(dataset_loader_np, flag_print=False):
    model_name = 'gradient_boosting_regressor'
    X_train, y_train, X_test, y_test = dataset_loader_np[0], dataset_loader_np[1], dataset_loader_np[2], \
                                       dataset_loader_np[3]

    model_name = 'gradient boosting regressor'
    gradient_boosting_regressor_model = GradientBoostingRegressor()
    gradient_boosting_regressor_model.fit(X_train, y_train)
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
    X_train, y_train, X_test, y_test = dataset_loader_np[0], dataset_loader_np[1], dataset_loader_np[2], \
                                       dataset_loader_np[3]

    svr_model = svm.SVR(C=param_c)
    svr_model.fit(X_train, y_train)
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
    dataset_loader_np = csv_to_dataset_np(csv_path, feature_str, non_normalization_feature)
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
    if model_name == 'ordinary_regression':
        return ordinary_regression(dataset_loader_np, print_flag)
    elif model_name == 'LASSO_regression':
        return LASSO_regression(dataset_loader_np, print_flag)
    elif model_name == 'randomforest_regressor':
        return randomforest_regressor(dataset_loader_np, print_flag)
    elif model_name == 'extratrees_regressor':
        return extratrees_regressor(dataset_loader_np, print_flag)
    elif model_name == 'gradient_boosting_regressor':
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


def csv_to_dataset_np(csv_path: str, feature_str: list, non_normalization_feature: list = None):
    import numpy as np
    import pandas as pd

    dt = pd.read_csv(csv_path)

    if type(dt['cbwd'][1]) is str:
        dt = dataPreprocess.data_conversion(dt)
    nan_index, non_nan_index = dataPreprocess.detect_missing_data(dt)

    train_dataset_X, train_dataset_y = [], []
    test_dataset_X, test_dataset_y = [], []

    all_dataset = dataPreprocess.standard_normalization(dt, feature_str, non_normalization_feature)

    for index, item in enumerate(non_nan_index):
        if index % 7 == 6:
            test_dataset_X.append(all_dataset.loc[item, feature_str].values)
            test_dataset_y.append(all_dataset.loc[item, 'pm2.5'])
        else:
            train_dataset_X.append(all_dataset.loc[item, feature_str].values)
            train_dataset_y.append(all_dataset.loc[item, 'pm2.5'])

    X_train, y_train = np.array(train_dataset_X), np.array(train_dataset_y)
    X_test, y_test = np.array(test_dataset_X), np.array(test_dataset_y)

    # tuple 防止对训练集和测试集进行更改
    dataset_loader_np = (X_train, y_train, X_test, y_test)

    return dataset_loader_np


if __name__ == "__main__":
    model_selection_list = ['ordinary_regression', 'LASSO_regression', 'randomforest_regressor',
                            'extratrees_regressor', 'gradient_boosting_regressor']
    csv_path = './new_feature.csv'
    feature_str = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir', 'feature 1']
    non_normalization_feature = ['cbwd', 'feature 1']
    model_evaluation(model_selection_list, csv_path, feature_str, non_normalization_feature)
