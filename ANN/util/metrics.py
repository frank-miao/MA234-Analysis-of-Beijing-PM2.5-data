import time
import torch
from thop import profile



def get_accuracy(net, features, labels,loss_function):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss_function(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


# 每一轮训练模型的好坏应该写入到一个文件中