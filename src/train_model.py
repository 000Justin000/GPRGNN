#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.

"""

"""

import argparse
from dataset_utils import DataLoader
from utils import random_planetoid_splits, random_splits
from GNN_models import *

import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

import numpy as np
from sklearn.metrics import r2_score


def RunExp(args, dataset, data, Net):

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        loss = F.mse_loss(out, data.y[data.train_mask])
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        y_pred, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = y_pred[mask]
            acc = r2_score(data.y[mask].detach().cpu().numpy(), pred.detach().cpu().numpy())
            loss = F.mse_loss(pred, data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_splits
    data = permute_masks(data)

    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                      {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                      {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.lr}], lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    return test_acc, best_val_acc, Gamma_0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.025)
    parser.add_argument('--val_rate', type=float, default=0.025)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop', choices=['PPNP', 'GPR_prop'])

    parser.add_argument('--dataset', default='county_facebook_2016')
    parser.add_argument('--target', default='election')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN'], default='GPRGNN')

    args = parser.parse_args()

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'JKNet':
        Net = GCN_JKNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN

    dname = args.dataset
    tname = args.target
    dataset = DataLoader(dname, tname)
    data = dataset[0]

    RPMAX = args.RPMAX
    Init = args.Init

    Gamma_0 = None

    args.Gamma = Gamma_0

    acc_list = []
#   for alpha_c in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 0.999]:
    for alpha_c in [0.00, 0.50, 0.90]:
        alpha = 1.0 - alpha_c
        acc_at_alpha = []
#       for _ in range(RPMAX):
        for _ in range(2):
            test_acc, best_val_acc, Gamma_0 = RunExp(args, dataset, data, Net)
            acc_at_alpha.append(test_acc)
        acc_list.append(sum(acc_at_alpha) / len(acc_at_alpha))
    print("overall accuracy:    ", max(acc_list), "    ", acc_list)
