from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

import validation

import pandas as pd
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

    random_forest_regressor_model = RandomForestRegressor(max_depth=13, n_estimators=150).fit(X_train, y_train)
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


def mlp_regressor(dataset_loader_np, flag_print=False):
    model_name = 'mlp regressor'
    X_train, y_train, X_test, y_test = dataset_loader_np

    amm_model = MLPRegressor(hidden_layer_sizes=(16, 32, 64, 128, 64, 32, 16, 8))
    amm_model.fit(X_train, y_train)
    train_score = r2_score(y_train, amm_model.predict(X_train))
    test_score = r2_score(y_test, amm_model.predict(X_test))

    if flag_print:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The R2 score of train_dataset is {0}".format(train_score))
        print("The R2 score of test_dataset is {0}".format(test_score))

    model_evaluation_result = (model_name, train_score, test_score)

    return model_evaluation_result


def lstm(dataset_loader_np, flag_print=False):
    model_name = 'lstm'
    X_train, y_train, X_test, y_test = dataset_loader_np

    X_train_lstm = np.zeros((X_train.shape[0] // 24 - 1,
                             24,
                             X_train.shape[-1]))
    y_train_lstm = np.zeros((y_train.shape[0] // 24 - 1,
                             24))

    train_rows = range(0, X_train.shape[0] // 24 - 1)

    for i, row in enumerate(train_rows):
        X_train_lstm[i, :, :] = X_train[row: row + 24]
        y_train_lstm[i, :] = y_train[row + 24: row + 24 + 24]

    X_test_lstm = np.zeros((X_test.shape[0] // 24 - 1,
                            24,
                            X_test.shape[-1]))
    y_test_lstm = np.zeros((y_test.shape[0] // 24 - 1,
                            24))

    test_rows = range(0, X_test.shape[0] // 24 - 1)

    for i, row in enumerate(test_rows):
        X_test_lstm[i, :, :] = X_test[row: row + 24]
        y_test_lstm[i, :] = y_test[row + 24: row + 24 + 24]

    lstm_model = Sequential()
    lstm_model.add(LSTM(32,
                        input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[-1]),
                        return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(32, return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(32))
    lstm_model.add(Dropout(0.2))

    lstm_model.add(Dense(24))

    lstm_model.compile(loss='mse',
                       optimizer='rmsprop')
    lstm_model.summary()

    lstm_model.fit(X_train_lstm, y_train_lstm,
                   epochs=60,
                   batch_size=128,
                   validation_data=(X_test_lstm, y_test_lstm))

    train_score = r2_score(y_train_lstm, lstm_model.predict(X_train_lstm))
    test_score = r2_score(y_test_lstm, lstm_model.predict(X_test_lstm))

    if flag_print:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The R2 score of train_dataset is {0}".format(train_score))
        print("The R2 score of test_dataset is {0}".format(test_score))

    model_evaluation_result = (model_name, train_score, test_score)

    return model_evaluation_result


if __name__ == "__main__":
    '''
    Models include ordinary regression, LASSO regression, Random Forest regressor, Extra trees regressor, \
    Gradient Boosting regressor, SVR, MLP regression
    '''

    model_selection_list = ['ordinary regression', 'LASSO regression', 'random forest regressor',
                            'extra trees regressor', 'gradient boosting regressor', 'mlp regressor']
    csv_path = 'impute_files/interpolate.csv'
    file_list = ['interpolate.csv', 'knn.csv', 'mean.csv', 'median.csv', 'mode.csv', 'zero.csv']
    feature_str = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']
    non_normalization_feature = ['cbwd']
    for item in file_list:
        validation.model_evaluation(model_selection_list, 'impute_files/' + item, feature_str,
                                    non_normalization_feature,
                                    task='Regression', plot_flag=True)
