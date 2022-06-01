import copy

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso

from data_preprocess import data_conversion
from data_preprocess import detect_missing_data


def AIC_feature_selection(data, selected_feature: list):
    X, y = load_data(data, selected_feature)
    X = sm.add_constant(X)
    selected_variable_aic = []
    for i in range(len(selected_feature)):
        aic_value = []
        alternative_var = []
        temp_var = copy.copy(selected_variable_aic)
        for var in selected_feature:
            if var in selected_variable_aic:
                continue

            temp_var.append(var)
            X_temp = X.loc[:, temp_var]
            lr = sm.OLS(y, X_temp).fit()
            aic_value.append(lr.aic)
            alternative_var.append(var)

        if len(aic_value) != 0:
            selected_variable_aic.append(alternative_var[aic_value.index(min(aic_value))])

    return selected_variable_aic[::-1]


def BIC_feature_selection(data, selected_feature: list):
    X, y = load_data(data, selected_feature)
    X = sm.add_constant(X)
    selected_variable_bic = []
    for i in range(len(selected_feature)):
        bic_value = []
        alternative_var = []
        temp_var = copy.copy(selected_variable_bic)
        for var in selected_feature:
            if var in selected_variable_bic:
                continue

            temp_var.append(var)
            X_temp = X.loc[:, temp_var]
            lr = sm.OLS(y, X_temp).fit()
            bic_value.append(lr.bic)
            alternative_var.append(var)

        if len(bic_value) != 0:
            selected_variable_bic.append(alternative_var[bic_value.index(min(bic_value))])

    return selected_variable_bic[::-1]


def LASSO_feature_selection(data, selected_feature: list):
    X, y = load_data(data, selected_feature)
    selected_variable_lasso = []

    for i in range(150, 0, -1):
        lasso = Lasso(alpha=i).fit(X, y)
        if set(X.columns[np.abs(lasso.coef_) > 1e-5]) == set(selected_variable_lasso):
            continue
        selected_variable_lasso.extend(
            [i for i in X.columns[np.abs(lasso.coef_) > 1e-5] if i not in selected_variable_lasso])
    return selected_variable_lasso


def RF_feature_selection(data, selected_feature: list):
    X, y = load_data(data, selected_feature)
    rf = RandomForestRegressor().fit(X, y)
    feature_importance = rf.feature_importances_
    arrIndex = np.array(feature_importance).argsort()
    return np.array(selected_feature)[arrIndex[::-1]]


def Mutual_Information_feature_selection(data, selected_feature: list):
    X, y = load_data(data, features)
    mi = mutual_info_regression(X, y)
    arrIndex = np.array(mi).argsort()
    return np.array(selected_feature)[arrIndex[::-1]]


def PCA_feature_extraction(data, selected_feature: list, n_components=4, kernel=False):
    X, y = load_data(data, selected_feature)
    if kernel:
        pca = KernelPCA(n_components=4).fit(X)
    else:
        pca = PCA(n_components=n_components).fit(X)
    X = pca.transform(X)
    return X, y


def load_data(data: pd.DataFrame, selected_feature: list):
    data = data_conversion(data)
    nan_index, non_nan_index = detect_missing_data(data)
    X = data.loc[non_nan_index, selected_feature]
    y = data.loc[non_nan_index, 'pm2.5']
    return X, y


if __name__ == '__main__':
    features = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir', 'feature 1']
    print(Mutual_Information_feature_selection(pd.read_csv('new_feature.csv'), features))
