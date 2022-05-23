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


def data_conversion(data:pd.DataFrame):
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


def regression_dataloader(csv_path: str, selected_feature: list, non_normalization_feature: list = None):
    dt = pd.read_csv(csv_path)

    if type(dt['cbwd'][1]) is str:
        dt = data_conversion(dt)
    nan_index, non_nan_index = detect_missing_data(dt)

    train_dataset_X, train_dataset_y = [], []
    test_dataset_X, test_dataset_y = [], []

    all_dataset = standard_normalization(dt, selected_feature, non_normalization_feature)

    for index, item in enumerate(non_nan_index):
        if index % 7 == 6:
            test_dataset_X.append(all_dataset.loc[item, selected_feature].values)
            test_dataset_y.append(all_dataset.loc[item, 'pm2.5'])
        else:
            train_dataset_X.append(all_dataset.loc[item, selected_feature].values)
            train_dataset_y.append(all_dataset.loc[item, 'pm2.5'])

    X_train, y_train = np.array(train_dataset_X), np.array(train_dataset_y)
    X_test, y_test = np.array(test_dataset_X), np.array(test_dataset_y)

    # tuple 防止对训练集和测试集进行更改
    dataset_loader_np = (X_train, y_train, X_test, y_test)

    return dataset_loader_np


if __name__ == '__main__':
    pass
