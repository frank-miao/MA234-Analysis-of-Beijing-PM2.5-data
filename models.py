import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import svm
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


def svr(dataset_loader_np, flag_print=False,param_c = 1.2):
    model_name = 'svr'
    X_train, y_train, X_test, y_test = dataset_loader_np[0], dataset_loader_np[1], dataset_loader_np[2], \
                                       dataset_loader_np[3]

    svr_model =svm.SVR(C=1.2)
    svr_model.fit(X_train,y_train)
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
def model_evaluation(model_selection_list: list, csv_path, feature_str: list,print_flag = False):
    dataset_loader_np = csv_to_dataset_np(csv_path, feature_str)
    all_model_evaluation_result = []
    for item in tqdm(model_selection_list):
        single_model_evaluation_result = model_call(item,dataset_loader_np,print_flag)
        all_model_evaluation_result.append(single_model_evaluation_result)

    # print part
    for item in all_model_evaluation_result:
        print("The model is {0} ".format(item[0]),end="")
        print("train_r2 is {0} ".format(item[1]),end="")
        print("test_r2 is {0}".format(item[2]))


def model_call(model_name: str,dataset_loader_np,print_flag = False):
    if model_name == 'ordinary_regression':
        return ordinary_regression(dataset_loader_np,print_flag)
    elif model_name == 'LASSO_regression':
        return LASSO_regression(dataset_loader_np,print_flag)
    elif model_name == 'randomforest_regressor':
        return randomforest_regressor(dataset_loader_np,print_flag)
    elif model_name == 'extratrees_regressor':
        return extratrees_regressor(dataset_loader_np,print_flag)
    elif model_name == 'gradient_boosting_regressor':
        return gradient_boosting_regressor(dataset_loader_np,print_flag)
    elif model_name == 'svr':
        return svr(dataset_loader_np, print_flag)




def csv_to_dataset_np(csv_path: str, feature_str: list):
    import numpy as np
    import pandas as pd


    dt = pd.read_csv(csv_path)

    dt = dt.dropna()

    X_pd = dt.loc[:, feature_str]  # 'year','month','day','hour',
    # X_pd = dt.loc[:, ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']]
    Y_pd = dt.loc[:, 'pm2.5']

    # normalization
    X_norm_pd = data_normalization(X_pd)

    X = X_norm_pd.values.astype(float)
    Y = Y_pd.values.astype(int)

    Y = np.reshape(Y, (-1, 1))

    X_test, y_test = X[[i for i in range(X.shape[0]) if i % 7 == 6]][:, 0:], Y[[i for i in range(Y.shape[0]) if
                                                                                i % 7 == 6]][:, 0]
    X_train, y_train = X[[i for i in range(X.shape[0]) if i % 7 != 6]][:, 0:], Y[[i for i in range(Y.shape[0]) if
                                                                                  i % 7 != 6]][:, 0]

    # tuple 防止对训练集和测试集进行更改
    dataset_loader_np = (X_train, y_train, X_test, y_test)

    return dataset_loader_np


def data_normalization(data):
    # 获取所有数值型数据 注意 这里目前传递的是引用，非其它数据类型
    numeric_features = data.dtypes[data.dtypes != 'object'].index

    data[numeric_features] = data[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    # 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
    data[numeric_features] = data[numeric_features].fillna(0)

    return data


if __name__ == "__main__":
    model_selection_list = ['ordinary_regression','LASSO_regression','randomforest_regressor',
                            'extratrees_regressor','gradient_boosting_regressor','svr']
    csv_path = './new_feature.csv'
    test = pd.read_csv(csv_path)
    feature_str = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir','feature1','feature2']
    model_evaluation(model_selection_list,csv_path,feature_str)

