
import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from .preprocess import data_normalization

def csv_to_dataloader(csv_path):

    dt = pd.read_csv(csv_path)

    X_pd = dt.loc[:,['DEWP','TEMP','PRES','cbwd','Iws','Is','Ir']] # 'year','month','day','hour',
    Y_pd = dt.loc[:,'pm2.5']

    # normalization
    X_norm_pd = data_normalization(X_pd)

    X = X_norm_pd.values.astype(float)
    Y = Y_pd.values.astype(int)

    Y = np.reshape(Y,(-1,1))


    # TODO:这里按照project要求进行抽取 已完成
    X_test, Y_test = X[[i for i in range(X.shape[0]) if i % 7 == 6]][:, 0:], Y[[i for i in range(Y.shape[0]) if i % 7 == 6]][:, 0]
    X_train, Y_train = X[[i for i in range(X.shape[0]) if i % 7 != 6]][:, 0:], Y[[i for i in range(Y.shape[0]) if i % 7 != 6]][:, 0]

    # X_train ,X_test,Y_train,Y_test =train_test_split(X,Y,test_size= 0.1,shuffle=True)

    X_train, Y_train = torch.FloatTensor(X_train),torch.FloatTensor(Y_train)
    X_test,  Y_test= torch.FloatTensor(X_test),torch.FloatTensor(Y_test)


    train_dataset =  torch.utils.data.TensorDataset(X_train,Y_train)
    # print(train_dataset)
    test_dataset = torch.utils.data.TensorDataset(X_test,Y_test)
    # print(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset =train_dataset ,batch_size = 10,shuffle =True)
    test_loader = torch.utils.data.DataLoader(dataset =test_dataset ,batch_size = 10,shuffle =True)
    # print(train_loader)

    return train_loader,test_loader

