from sklearn.model_selection import KFold
import data_preprocess
import classification_models
from tqdm import tqdm


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
                     plot_flag=False):
    # TODO: 仿照回归模型明天补完
    pass


if __name__ == '__main__':
    feature_str = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'feature 1']
    non_normalization_feature = ['cbwd', 'feature 1']
    dataloader = data_preprocess.classification_dataloader('new_feature.csv', selected_feature=feature_str,
                                                           non_normalization_feature=non_normalization_feature)
    classification_cv(dataloader, classification_models.svm_classification)
