import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy


def read_file(file_path):
    return pd.read_csv(file_path)


def standard_normalization(data: pd.DataFrame, selected_feature: list, non_normalization_feature: list = None):
    data_copy = copy.deepcopy(data)
    normalization_feature = selected_feature.copy()
    if non_normalization_feature:
        for item in non_normalization_feature:
            if item in normalization_feature:
                normalization_feature.remove(item)

    normalized_data = StandardScaler().fit_transform(data_copy[normalization_feature])
    data_copy[normalization_feature] = normalized_data
    return data_copy


def detect_missing_data(data: pd.DataFrame):
    pm25_data = data.loc[:, "pm2.5"].values.reshape(-1, 1)
    nan_index, _ = np.where(np.isnan(pm25_data))
    non_nan_index, _ = np.where(~np.isnan(pm25_data))
    return nan_index, non_nan_index


def data_conversion(data):
    data_copy = copy.deepcopy(data)
    cbwd_one_hot = dict(zip(set(data_copy['cbwd']), range(4)))
    cbwd_one_hot_inverse = dict(zip(range(4), set(data_copy['cbwd'])))

    X_cbwd = copy.deepcopy(data_copy['cbwd'].values)
    X_cbwd_new = np.asarray(X_cbwd)
    for i, item in enumerate(X_cbwd):
        X_cbwd_new[i] = cbwd_one_hot[item]

    data_copy['cbwd'] = X_cbwd_new
    return data_copy


def clear_missing_value(data, clear=False):
    data_copy = copy.deepcopy(data)
    if clear:
        nan_index, non_nan_index = detect_missing_data(data)
        data_copy = data_copy[non_nan_index]
    return data_copy


if __name__ == '__main__':
    pass
