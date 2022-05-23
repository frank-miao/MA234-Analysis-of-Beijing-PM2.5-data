import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from information_enum import PmState
import copy


def read_file(file_path) -> pd.DataFrame:
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
    nan_index, _ = np.where(np.isnan(pm25_data))  # numpy
    non_nan_index, _ = np.where(~np.isnan(pm25_data))  # numpy
    return nan_index, non_nan_index


def data_conversion(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = copy.deepcopy(data)
    cbwd_one_hot = dict(zip(set(data_copy['cbwd']), range(4)))
    cbwd_one_hot_inverse = dict(zip(range(4), set(data_copy['cbwd'])))

    X_cbwd = copy.deepcopy(data_copy['cbwd'].values)
    X_cbwd_new = np.asarray(X_cbwd)
    for i, item in enumerate(X_cbwd):
        X_cbwd_new[i] = cbwd_one_hot[item]

    data_copy['cbwd'] = X_cbwd_new
    return data_copy


def clear_missing_value(data: pd.DataFrame, clear=False) -> pd.DataFrame:
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


# divide the pm2.5 data to three state
def data_partition(data_pd: pd.DataFrame) -> pd.DataFrame:
    data_pd_copy = copy.deepcopy(data_pd)
    data_pd_copy['pm2.5'] = data_pd_copy['pm2.5'].apply(data_pm_partition_cal)
    # low_pm_state 1 polluting episode 2 very high PM 3
    return data_pd_copy

def data_pm_partition_cal(single_pm25):
    if single_pm25 <= PmState.PARTITION_BETWEEN_LOW_POLLUTING.value:
        return PmState.LOW_PM_STATE.value
    elif PmState.PARTITION_BETWEEN_LOW_POLLUTING.value < single_pm25 <= PmState.PARTITION_BETWEEN_POLLUTING_HIGH.value:
        return PmState.POLLUTING_EPISODE.value
    elif single_pm25 > PmState.PARTITION_BETWEEN_POLLUTING_HIGH.value:
        return PmState.VERY_HIGH_PM_STATE.value


def classification_data_loader(data_pd: pd.DataFrame) -> tuple:
    hour_map_dict = {
        0: 1,
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 3,
        7: 3,
        8: 3,
        9: 4,
        10: 4,
        11: 4,
        12: 5,
        13: 5,
        14: 5,
        15: 6,
        16: 6,
        17: 6,
        18: 7,
        19: 7,
        20: 7,
        21: 8,
        22: 8,
        23: 8,
    }
    cbwd_dict = dict(zip(set(data_pd['cbwd']), range(4)))
    data_pd_copy = copy.deepcopy(data_pd)
    data_pd_copy['hour'] = data_pd_copy['hour'].apply(lambda single_hour: hour_map_dict[single_hour])
    data_pd_copy['cbwd'] = data_pd_copy['cbwd'].apply(lambda x: cbwd_dict[x])
    data_pd_copy_groupby = data_pd_copy.groupby(
        [data_pd_copy['year'], data_pd_copy['month'], data_pd_copy['day'], data_pd_copy['hour']])
    pm25_series = data_pd_copy_groupby['pm2.5'].aggregate(handle_group_by)
    result_pd = data_pd_copy_groupby.mean()
    result_pd['pm2.5'] = pm25_series
    result_pd['cbwd'] = result_pd['cbwd'].apply(lambda single_cbwd: round(single_cbwd, 0))
    result_pd = data_partition(result_pd)
    result_pd = result_pd.dropna().reset_index()

    result_pd.drop('No', axis=1, inplace=True)


    X = result_pd.drop('pm2.5', axis=1)
    y = result_pd['pm2.5']
    X_y = (X, y)

    return X_y

def handle_group_by(data):
    n = 3
    sum = 0
    for i, item in enumerate(data):
        sum += item
        if np.isnan(item):
            n -= 1

    if n == 0:
        return np.nan
    else:
        return sum / n


if __name__ == '__main__':
    data_pd = read_file('./backup/PRSA_data_raw.csv')
    data_pd_2 = classification_data_loader(data_pd)
    data_pd_2.to_csv("testtter.csv")
