import argparse
import sys
import os
import os.path as osp

import numpy as np
from sympy import arg
import torch

from training.gnn_mdi import train_gnn_mdi
from uci.uci_subparser import add_uci_subparser
from utils.utils import auto_select_gpu
import pickle
from time import time
import warnings
warnings.filterwarnings('ignore')

import pickle
from numpy.core.numeric import NaN
import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing

import torch
import random
import numpy as np
import pdb
import math

from utils.utils import get_known_mask, mask_edge
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from uci.adj_matrix import *
# from training.GAugO_method import *
from sklearn.metrics.pairwise import cosine_similarity

# conf, cos, each_conf, lift
rules_attr_method = 'conf'
ratio = 0.3
initial = 'random'
mask_type = 'MCAR'

def create_node(df, mode, df_y):
    if mode == 0: # onehot feature node, all 1 sample node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol))
        feature_node[np.arange(ncol), feature_ind] = 1 # nrow x  ncol 
        sample_node = [[1]*ncol for i in range(nrow)] # nrow x  ncol
        node = sample_node + feature_node.tolist()  # nrow x  ncol 
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
    # create fully connected bidirectional graph
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

def get_edge_with_random(x, confidence, num_edge = None):
    # 对于NxN 的图 随机 generate N 那么多的边
    # 
    add_edge_start = []
    add_edge_end = []
    index = [i for i in range(x.shape[0])]  
    edge_index = []
    while(len(edge_index) < num_edge*2):
        edge = random.sample(index,2)
        if edge in edge_index:
            continue
        else:
            add_edge_start.append(edge[0])
            add_edge_end.append(edge[1])
            edge_index.append(edge)        
            edge_index.append([edge[1], edge[0]])
    return add_edge_start, add_edge_end

def get_data(df_X, df_y, node_mode, train_edge_prob, split_sample_ratio, split_by, train_y_prob, confidence, seed=0, dataset=None, mechanism="MCAR", normalize=True):
    if len(df_y.shape)==1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape)==2:
        df_y = df_y[0].to_numpy()
    features = pd.DataFrame(df_X.values)

    if normalize:
        x = df_X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)

    # created fully connected grpah
    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    node_init = create_node(df_X, node_mode, df_y) 
    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(df_y, dtype=torch.float)
    
    #set seed to fix known/unknwon edges
    torch.manual_seed(seed)
    # select missing mechanism
    # if mask_type == 'MAR' or mask_type == 'MNAR':
    #     X = simulate_nan(df_X.to_numpy(), 0.3, mask_type)
    #     train_mask = ~X['mask'].astype(bool)
    #     train_edge_mask = torch.from_numpy(train_mask.reshape(-1))
    # elif mask_type == 'MCAR':
    #     train_edge_mask = get_known_mask(train_edge_prob, int(edge_attr.shape[0]/2))

    # if mechanism == "MCAR":
    #     train_edge_mask = produce_NA(edge_attr[:int(edge_attr.shape[0]/2)], p_miss=1-train_edge_prob, mecha="MCAR", n_row=df_X.shape[0], n_col=df_X.shape[1])
    # elif mechanism == "MAR":
    #     train_edge_mask = produce_NA(edge_attr[:int(edge_attr.shape[0]/2)], p_miss=1-train_edge_prob, mecha="MAR", n_row=df_X.shape[0], n_col=df_X.shape[1], p_obs=0.5)
    # elif mechanism == "MNAR":
    #     train_edge_mask = produce_NA(edge_attr[:int(edge_attr.shape[0]/2)], p_miss=1-train_edge_prob, mecha="MNAR", n_row=df_X.shape[0], n_col=df_X.shape[1], opt="quantile", p_obs=0.5, q=0.3)
    # else:
    #     raise ValueError("Missing mechanism not implemented")
    
    train_edge_mask = ~torch.isnan(edge_attr[:int(edge_attr.shape[0] / 2)]).reshape(1, -1)[0]
    # print(train_edge_mask)
    edge_attr[torch.isnan(edge_attr)] = 0

    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0)
    #mask edges based on the generated train_edge_mask
    #train_edge_index is known, test_edge_index in unknwon, i.e. missing
    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr,
                                                double_train_edge_mask, True)
    
    
    random_edges = np.floor(1.0 * len(features))
    add_edge_start, add_edge_end = get_edge_with_random(features,confidence,random_edges)
        
    edge_start = (train_edge_index[0][0:int(train_edge_index[0].shape[0]/2)]).numpy().tolist() + add_edge_start
    edge_end = (train_edge_index[1][0:int(train_edge_index[1].shape[0]/2)]).numpy().tolist() + add_edge_end
    edge_start_ = edge_start + edge_end
    edge_end_ = edge_end + edge_start
    train_edge_index_ = torch.tensor([edge_start_, edge_end_], dtype=int)
    obob_edge_start = add_edge_start+add_edge_end
    obob_edge_end = add_edge_end+add_edge_start
    obob_edge_index = torch.tensor([obob_edge_start,obob_edge_end])
    

    obob_adj_train = get_obob_adj_matrix(df_X, obob_edge_index)
    obob_adj_orig = scipysp_to_pytorchsp(obob_adj_train).to_dense()  # 边的矩阵的torch.tensor表示
    obob_adj_norm = normalize_adj_(obob_adj_train)

    train_labels = train_edge_attr[:int(train_edge_attr.shape[0]/2),0]
    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr,
                                                ~double_train_edge_mask, True)
    test_labels = test_edge_attr[:int(test_edge_attr.shape[0]/2),0]
    #mask the y-values during training, i.e. how we split the training and test sets
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
            user_num=df_X.shape[0],
            obob_edge_index = obob_edge_index,
            # obob_adj_train = obob_adj_train,
            obob_adj_norm = obob_adj_norm,
            obob_adj_orig=obob_adj_orig,
            mask=train_edge_mask
            )

        
    return data

def load_data(args, xmiss):
    # uci_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    # df_np = np.loadtxt(uci_path+'/raw_data/{}/data/data.txt'.format(args.data))
    # df_y = pd.DataFrame(df_np[:, -1:])
    # df_X = pd.DataFrame(df_np[:, :-1])

    df_X = xmiss.copy()
    df_y = pd.Series(np.zeros(df_X.shape[0]))

    # print(df_X, df_y)

    if args.data in ['concrete','protein','power','wine','heart','DOW30','diabetes']:
        confidence = 0.6
    elif args.data in ['ecommerce']:
        confidence = 0.5
    elif args.data in ['housing']:
        confidence = 0.7
    else:
        confidence = 0.5

    data = get_data(df_X, df_y, args.node_mode, args.train_edge, args.split_sample, args.split_by, args.train_y, confidence, args.seed, args.data, args.missing_mechanism)
    return data




def igrm_impute(xmiss):
    parser = argparse.ArgumentParser()# domain
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--domain', type=str, default='uci')
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
    parser.add_argument('--node_mode', type=int, default=0)  # 0: feature onehot, sample all 1; 1: all onehot
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
    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--log_dir', type=str, default='0')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--save_prediction', action='store_true', default=False)
    parser.add_argument('--transfer_dir', type=str, default=None)
    parser.add_argument('--transfer_extra', type=str, default='')
    parser.add_argument('--mode', type=str, default='train') # debug
    parser.add_argument('--data', type=str, default='housing')
    parser.add_argument('--missing_mechanism', type=str, default='MCAR')
    
    parser.add_argument('--train_edge', type=float, default=0.8)
    parser.add_argument('--split_sample', type=float, default=0.)
    parser.add_argument('--split_by', type=str, default='y') # 'y', 'random'
    parser.add_argument('--split_train', action='store_true', default=False)
    parser.add_argument('--split_test', action='store_true', default=False)
    parser.add_argument('--train_y', type=float, default=0.8)

    parser.add_argument('--method', type=str, default='mean')
    parser.add_argument('--level', type=int, default=0)

    parser.add_argument('--best_level', action='store_true', default=False)
    parser.add_argument('--comment', type=str, default='v1')

    
    subparsers = parser.add_subparsers()
    add_uci_subparser(subparsers)
    args = parser.parse_args()
    # print(args)

    # select device
    if torch.cuda.is_available():
        cuda = auto_select_gpu()
        cuda = 0
        device = torch.device('cuda:{}'.format(cuda))
    else:
        # print('Using CPU')
        device = torch.device('cpu')

    seeds = [0,1314,100,1000,2000]
    

    for i in range(0,1):
        start = time()
        seed = seeds[i]
        args.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # print("start load data")

        data = load_data(args, xmiss)

        cmd_input = 'python ' + ' '.join(sys.argv) + ' ' + str(seed) + '\n'

        # print("start training")
        pred = train_gnn_mdi(data, args, device)

        # print(pred)

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
        
        
