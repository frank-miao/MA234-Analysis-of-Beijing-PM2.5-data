import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor

import data_preprocess
import regression_models


def mean_value_impute(data: pd.DataFrame, save=False) -> pd.DataFrame:
    data_mean = copy.deepcopy(data)
    pm25 = data_mean.loc[:, "pm2.5"].values.reshape(-1, 1)

    pm25_mean = SimpleImputer()
    pm25_mean = pm25_mean.fit_transform(pm25)

    data_mean.loc[:, "pm2.5"] = pm25_mean
    if save:
        data_mean.to_csv('./impute_files/mean.csv')
    return data_mean


def median_value_impute(data: pd.DataFrame, save=False) -> pd.DataFrame:
    data_median = copy.deepcopy(data)
    pm25 = data_median.loc[:, "pm2.5"].values.reshape(-1, 1)
    pm25_median = SimpleImputer(strategy="median")  # 使用中位数填补
    pm25_median = pm25_median.fit_transform(pm25)
    data_median.loc[:, "pm2.5"] = pm25_median
    if save:
        data_median.to_csv('./impute_files/median.csv')
    return data_median


def zero_value_impute(data: pd.DataFrame, save=False) -> pd.DataFrame:
    data_zero = copy.deepcopy(data)
    data_zero.fillna(0, inplace=True)
    if save:
        data_zero.to_csv('./impute_files/zero.csv')
    return data_zero


def mode_value_impute(data: pd.DataFrame, save=False) -> pd.DataFrame:
    data_mode = copy.deepcopy(data)
    pm25 = data_mode.loc[:, "pm2.5"].values.reshape(-1, 1)
    pm25_mode = SimpleImputer(strategy="most_frequent")  # 使用中位数填补
    pm25_mode = pm25_mode.fit_transform(pm25)
    data_mode.loc[:, "pm2.5"] = pm25_mode
    if save:
        data_mode.to_csv('./impute_files/mode.csv')
    return data_mode


def KNN_impute(data: pd.DataFrame, save=False) -> pd.DataFrame:
    nan_index, non_nan_index = data_preprocess.detect_missing_data(data)
    x = data.iloc[:, 6:].values[non_nan_index]
    y = data.iloc[:, 5].values[non_nan_index]
    KNR = KNeighborsRegressor(n_neighbors=5).fit(x, y)
    data_knn = copy.deepcopy(data)
    data_knn.loc[nan_index, 'pm2.5'] = KNR.predict(data.iloc[:, 6:].values[nan_index])
    if save:
        data_knn.to_csv('./impute_files/knn.csv')
    return data_knn


def interpolate_impute(data: pd.DataFrame, save=False) -> pd.DataFrame:
    nan_index, non_nan_index = data_preprocess.detect_missing_data(data)
    x = data.No.values[non_nan_index]
    y = data['pm2.5'].values[non_nan_index]
    f = interpolate.interp1d(x, y, fill_value="extrapolate")
    y_new = f(data.No.values[nan_index])
    data_interpolation = copy.deepcopy(data)
    data_interpolation.loc[nan_index, 'pm2.5'] = y_new
    if save:
        data_interpolation.to_csv('./impute_files/interpolate.csv')
    return data_interpolation


if __name__ == '__main__':
    pass
