import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from training.gnn_mdi import train_gnn_mdi
from utils.utils import auto_select_gpu


import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing

import torch
import random
import numpy as np
import pdb

from utils.utils import get_known_mask, mask_edge

def create_node(df, mode):
    if mode == 0: # onehot feature node, all 1 sample node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol))
        feature_node[np.arange(ncol), feature_ind] = 1
        sample_node = [[1]*ncol for i in range(nrow)]
        node = sample_node + feature_node.tolist()
    elif mode == 1: # onehot sample and feature node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol+1))
        feature_node[np.arange(ncol), feature_ind+1] = 1
        sample_node = np.zeros((nrow,ncol+1))
        sample_node[:,0] = 1
        node = sample_node.tolist() + feature_node.tolist()
    return node

def create_edge(df):
    n_row, n_col = df.shape
    edge_start = []
    edge_end = []
    for x in range(n_row):
        edge_start = edge_start + [x] * n_col # obj
        edge_end = edge_end + list(n_row+np.arange(n_col)) # att    
    edge_start_new = edge_start + edge_end
    edge_end_new = edge_end + edge_start
    return (edge_start_new, edge_end_new)

def create_edge_attr(df):
    nrow, ncol = df.shape
    edge_attr = []
    for i in range(nrow):
        for j in range(ncol):
            edge_attr.append([float(df.iloc[i,j])])
    edge_attr = edge_attr + edge_attr
    return edge_attr

def get_data(df_X, df_y, node_mode, train_edge_prob, split_sample_ratio, split_by, train_y_prob, seed=0, mechanism='MCAR', normalize=True):
    if len(df_y.shape)==1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape)==2:
        df_y = df_y[0].to_numpy()

    if normalize:
        x = df_X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)
    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    node_init = create_node(df_X, node_mode) 
    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(df_y, dtype=torch.float)
    
    #set seed to fix known/unknwon edges
    torch.manual_seed(seed)
    # Introduce missing data
    
    # train_edge_mask = torch.where(torch.isnan(edge_attr[:int(edge_attr.shape[0] / 2)]))[0]
    train_edge_mask = ~torch.isnan(edge_attr[:int(edge_attr.shape[0] / 2)]).reshape(1, -1)[0]
    # print(train_edge_mask)
    edge_attr[torch.isnan(edge_attr)] = 0

    # print(train_edge_mask, torch.sum(train_edge_mask))
    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0)

    #mask edges based on the generated train_edge_mask
    #train_edge_index is known, test_edge_index in unknwon, i.e. missing
    # print("start remove edge")
    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr,
                                                double_train_edge_mask, True)
    train_labels = train_edge_attr[:int(train_edge_attr.shape[0]/2),0]
    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr,
                                                ~double_train_edge_mask, True)
    test_labels = test_edge_attr[:int(test_edge_attr.shape[0]/2),0]
    #mask the y-values during training, i.e. how we split the training and test sets
    # train_y_mask = get_known_mask(train_y_prob, y.shape[0])

    train_y_mask = get_known_mask(train_y_prob, y.shape[0])
    
    test_y_mask = ~train_y_mask

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
            train_y_mask=train_y_mask, test_y_mask=test_y_mask,
            train_edge_index=train_edge_index,train_edge_attr=train_edge_attr,
            train_edge_mask=train_edge_mask,train_labels=train_labels,
            test_edge_index=test_edge_index,test_edge_attr=test_edge_attr,
            test_edge_mask=~train_edge_mask,test_labels=test_labels, 
            df_X=df_X,df_y=df_y,
            edge_attr_dim=train_edge_attr.shape[-1],
            user_num=df_X.shape[0]
            )
        
    return data

def load_data(args, xmiss):
    
    df_X = xmiss.copy()
    df_X = pd.DataFrame(df_X)

    df_y = pd.Series(np.zeros(df_X.shape[0]))



    data = get_data(df_X, df_y, args.node_mode, args.train_edge, args.split_sample, args.split_by, args.train_y, args.seed, args.mechanism)

    return data

def grape_impute(xmiss):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default=None,) # default to be 1 hidden of node_dim
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
    parser.add_argument('--aggr', type=str, default='mean',)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight; 1: as input to mlp
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--impute_hiddens', type=str, default='64')
    parser.add_argument('--impute_activation', type=str, default='relu')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--known', type=float, default=0.8) # 1 - edge dropout rate
    parser.add_argument('--auto_known', action='store_true', default=False)
    parser.add_argument('--loss_mode', type=int, default = 0) # 0: loss on all train edge, 1: loss only on unknown train edge
    parser.add_argument('--valid', type=float, default=0.) # valid-set ratio
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='2')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--save_prediction', action='store_true', default=False)
    parser.add_argument('--transfer_dir', type=str, default=None)
    parser.add_argument('--transfer_extra', type=str, default='')
    parser.add_argument('--mode', type=str, default='train') # debug
    parser.add_argument('--mechanism', type=str, default='MCAR') # debug
    
    parser.add_argument('--domain', type=str, default='uci')
    parser.add_argument('--data', type=str, default='housing')
    parser.add_argument('--train_edge', type=float, default=0.7)
    parser.add_argument('--split_sample', type=float, default=0.)
    parser.add_argument('--split_by', type=str, default='y') # 'y', 'random'
    parser.add_argument('--split_train', action='store_true', default=False)
    parser.add_argument('--split_test', action='store_true', default=False)
    parser.add_argument('--train_y', type=float, default=0.7)
    parser.add_argument('--node_mode', type=int, default=0)  # 0: feature onehot, sample all 1; 1: all onehot

    args = parser.parse_args()

    device = torch.device('cuda:0')

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = load_data(args, xmiss)

    # print("start training....")
    pred = train_gnn_mdi(data, args, device)


    cnt = 0
    for i in range(xmiss.shape[0]):
        for j in range(xmiss.shape[1]):
            if np.isnan(xmiss.iloc[i, j]):
                # print(i, j)
                # print(i, j, data.test_edge_index[0][cnt], data.test_edge_index[1][cnt]-xmiss.shape[0])
                if data.test_edge_index[0][cnt] == i and data.test_edge_index[1][cnt]-xmiss.shape[0] == j:
                    xmiss.iloc[i, j] = pred[cnt]
                cnt+=1

    return xmiss.values
