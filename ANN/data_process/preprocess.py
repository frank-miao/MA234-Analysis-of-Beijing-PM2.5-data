# 计算所有数据的均值和标准差
# 离散值处理
import pandas as pd

def data_normalization(data):

    # 获取所有数值型数据 注意 这里目前传递的是引用，非其它数据类型
    numeric_features = data.dtypes[data.dtypes != 'object'].index

    data[numeric_features] = data[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    # 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
    data[numeric_features] = data[numeric_features].fillna(0)

    return data


# 将传入的数据的指定列转换成onehot编码
def data_trans_onehot():

    pass