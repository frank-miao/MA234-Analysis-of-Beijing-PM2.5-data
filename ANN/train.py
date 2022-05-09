from data_process.dataloader import csv_to_dataloader
from util.metrics import *

import argparse
import os
import torch

from torch import nn

from models.ANN import ANN

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint-dir', type=str, default="./checkpoints/5_3_19_37")
parser.add_argument('--last-checkpoint', type=str, default=None)

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int, required=True)

args = parser.parse_args()


def train(model, train_loader, test_loader, optimizer, loss_fn):

    for epoch in range(args.epoch_start, args.epoch_end):
        print(f"Epoch {epoch}\n-------------------------------")
        # size = len(train_loader.dataset)
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):

            X, y = X.to(args.device), y.to(args.device)

            # Compute prediction error
            pred_y = model(X)
            loss = loss_fn(pred_y, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # if batch_idx % 100 == 0:
            #     loss, current = loss.item(), batch_idx * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        accuracy = get_accuracy(model, X, y, loss_fn)
        print("Accuracy: %.3f}" % accuracy)

        torch.save(model.state_dict(), args.checkpoint_dir + f'epoch-{epoch}.pth')


if __name__ == "__main__":

    csv_path = "datasets/PRSA_data_dropna.csv"

    # 得到dataloader
    train_loader, test_loader = csv_to_dataloader(csv_path)

    model = ANN().to(device=args.device)

    if args.last_checkpoint is not None:
        model.load_state_dict(torch.load(args.last_checkpoint, map_location=args.device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    loss_fn = nn.MSELoss()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train(model, train_loader, test_loader, optimizer, loss_fn)
