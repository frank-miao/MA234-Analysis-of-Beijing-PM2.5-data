from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import validation


def KNN_classification(dataloader, print_flag=False):
    model_name = 'KNN'
    X_train, y_train, X_test, y_test = dataloader
    neigh = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
    f1_train = f1_score(y_train, neigh.predict(X_train), average='micro')
    f1_test = f1_score(y_test, neigh.predict(X_test), average='micro')
    if print_flag:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The f1 score of train_dataset is {0}".format(f1_train))
        print("The f1 score of test_dataset is {0}".format(f1_test))

    model_evaluation_result = (model_name, f1_train, f1_test)

    return model_evaluation_result


def svm_classification(dataloader, print_flag=False):
    model_name = 'SVM'
    X_train, y_train, X_test, y_test = dataloader
    clf = svm.SVC(kernel='rbf').fit(X_train, y_train)
    f1_train = f1_score(y_train, clf.predict(X_train), average='micro')
    f1_test = f1_score(y_test, clf.predict(X_test), average='micro')
    if print_flag:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The f1 score of train_dataset is {0}".format(f1_train))
        print("The f1 score of test_dataset is {0}".format(f1_test))

    model_evaluation_result = (model_name, f1_train, f1_test)

    return model_evaluation_result


def decision_tree_classification(dataloader, print_flag=False):
    model_name = 'decision tree'
    X_train, y_train, X_test, y_test = dataloader
    clf = DecisionTreeClassifier(max_depth=10).fit(X_train, y_train)
    f1_train = f1_score(y_train, clf.predict(X_train), average='micro')
    f1_test = f1_score(y_test, clf.predict(X_test), average='micro')
    if print_flag:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The f1 score of train_dataset is {0}".format(f1_train))
        print("The f1 score of test_dataset is {0}".format(f1_test))

    model_evaluation_result = (model_name, f1_train, f1_test)

    return model_evaluation_result


def logistic_regression_classification(dataloader, print_flag=False):
    model_name = 'logistic regression'
    X_train, y_train, X_test, y_test = dataloader
    clf = LogisticRegression().fit(X_train, y_train)
    f1_train = f1_score(y_train, clf.predict(X_train), average='micro')
    f1_test = f1_score(y_test, clf.predict(X_test), average='micro')
    if print_flag:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The f1 score of train_dataset is {0}".format(f1_train))
        print("The f1 score of test_dataset is {0}".format(f1_test))

    model_evaluation_result = (model_name, f1_train, f1_test)

    return model_evaluation_result


def LDA_classification(dataloader, print_flag=False):
    model_name = 'LDA'
    X_train, y_train, X_test, y_test = dataloader
    clf = LinearDiscriminantAnalysis().fit(X_train, y_train)
    f1_train = f1_score(y_train, clf.predict(X_train), average='micro')
    f1_test = f1_score(y_test, clf.predict(X_test), average='micro')
    if print_flag:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The f1 score of train_dataset is {0}".format(f1_train))
        print("The f1 score of test_dataset is {0}".format(f1_test))

    model_evaluation_result = (model_name, f1_train, f1_test)

    return model_evaluation_result


def mlp_classification(dataloader, print_flag=False):
    model_name = 'mlp classification'
    X_train, y_train, X_test, y_test = dataloader

    mlp_model = MLPClassifier(hidden_layer_sizes=(16, 32, 64, 32, 16, 8))
    mlp_model.fit(X_train, y_train)
    f1_train = f1_score(y_train, mlp_model.predict(X_train), average='micro')
    f1_test = f1_score(y_test, mlp_model.predict(X_test), average='micro')

    if print_flag:
        # 打印当前的模型信息
        print("The model is {0}".format(model_name))
        print("The f1 score of train_dataset is {0}".format(f1_train))
        print("The f1 score of test_dataset is {0}".format(f1_test))

    model_evaluation_result = (model_name, f1_train, f1_test)

    return model_evaluation_result


if __name__ == '__main__':
    '''
    Models include KNN, SVM, decision tree, logistic regression, LDA
    '''

    model_selection_list = ['KNN', 'SVM', 'decision tree', 'logistic regression', 'LDA', 'mlp classification']
    csv_path = './new_feature.csv'
    feature_str = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'feature 1']
    non_normalization_feature = ['cbwd', 'feature 1']
    validation.model_evaluation(model_selection_list, csv_path, feature_str, non_normalization_feature,
                                task='classification')
