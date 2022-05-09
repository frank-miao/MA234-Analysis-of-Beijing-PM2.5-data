from data_process.dataloader import csv_to_dataloader
from util.metrics import *
import argparse
import torch

from torchvision import datasets, transforms
from models.ANN import ANN

# --------------- Arguments ---------------

parser = argparse.ArgumentParser()

parser.add_argument('--best-checkpoint', type=str)

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--batch-size', type=int, default=64)

args = parser.parse_args()

# 使用之前抽取的训练集和测试集



if __name__ == '__main__':


    csv_path = "datasets/PRSA_data_dropna.csv"


    # 得到dataloader
    train_loader, test_loader = csv_to_dataloader(csv_path)

    model = ANN().to(device=args.device)

    model.load_state_dict(torch.load(args.best_checkpoint, map_location=args.device))

    model.eval()

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device=args.device), y.to(device=args.device)
            pred_y = model(X)
            real_y = y
            print("pred y is",pred_y,"real y is",real_y)




