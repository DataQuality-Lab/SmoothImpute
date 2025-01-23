import csv
import numpy as np
from sklearn import model_selection, preprocessing

from ginn import GINN
from ginn.utils import degrade_dataset, data2onehot
import warnings
warnings.filterwarnings("ignore")

import argparse

import sys
import os
# 获取上一级目录的路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from utils_general import load_data_general
from utils_general import RMSE, MAE

from time import time

def main(p_miss=0.5, dataset="wine", mode="MCAR", rand_seed=1234):
    np.random.seed(rand_seed)

    start = time()
    
    data_x, xmiss, mask = load_data_general(data_name=dataset, miss_rate=p_miss, missing_mechanism=mode)
    
    y = np.zeros((data_x.shape[0],1),dtype='int')
    
    cat_cols = []
    num_cols = [i for i in range(data_x.shape[1])]
    X = xmiss.copy()
    y = np.reshape(y,-1)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, shuffle=False
    )
    # cx_train, cx_train_mask = degrade_dataset(x_train, missingness,seed, np.nan)
    # cx_test,  cx_test_mask  = degrade_dataset(x_test, missingness,seed, np.nan)
    cx_train = x_train.copy()
    cx_train_mask = np.where(np.isnan(cx_train), 0, 1)
    cx_test = x_test.copy()
    cx_test_mask = np.where(np.isnan(cx_test), 0, 1)

    cx_tr = np.c_[cx_train, y_train]
    cx_te = np.c_[cx_test, y_test]

    mask_tr = np.c_[cx_train_mask, np.ones(y_train.shape)]
    mask_te = np.c_[cx_test_mask,  np.ones(y_test.shape)]

    [oh_x, oh_mask, oh_num_mask, oh_cat_mask, oh_cat_cols] = data2onehot(
            np.r_[cx_tr,cx_te], np.r_[mask_tr,mask_te], num_cols, cat_cols
    )

    oh_x_tr = oh_x[:x_train.shape[0],:]
    oh_x_te = oh_x[x_train.shape[0]:,:]

    oh_mask_tr = oh_mask[:x_train.shape[0],:]
    oh_num_mask_tr = oh_mask[:x_train.shape[0],:]
    oh_cat_mask_tr = oh_mask[:x_train.shape[0],:]

    oh_mask_te = oh_mask[x_train.shape[0]:,:]
    oh_num_mask_te = oh_mask[x_train.shape[0]:,:]
    oh_cat_mask_te = oh_mask[x_train.shape[0]:,:]

    scaler_tr = preprocessing.MinMaxScaler()
    oh_x_tr = scaler_tr.fit_transform(oh_x_tr)

    scaler_te = preprocessing.MinMaxScaler()
    oh_x_te = scaler_te.fit_transform(oh_x_te)


    imputer = GINN(oh_x_tr,
                oh_mask_tr,
                oh_num_mask_tr,
                oh_cat_mask_tr,
                oh_cat_cols,
                num_cols,
                cat_cols
                )

    imputer.fit()
    imputed_tr = scaler_tr.inverse_transform(imputer.transform())


    imputer.add_data(oh_x_te,oh_mask_te,oh_num_mask_te,oh_cat_mask_te)

    imputed_te = imputer.transform()
    imputed_te = scaler_te.inverse_transform(imputed_te[x_train.shape[0]:])

    # print(f"imputed_tr, imputed_tr.shape: {imputed_tr[0]}, {imputed_tr.shape}")
    # print(f"imputed_te, imputed_te.shape: {imputed_te[0]}, {imputed_te.shape}")
    # print(f"x_train, x_train.shape: {x_train[0]}, {x_train.shape}")
    # print(f"x_test, x_test.shape: {x_test[0]}, {x_test.shape}")
    # print(f"mask, mask.shape: {mask[0]}, {mask.shape}")
    # print(f"data_x, data_x.shape: {data_x[0]}, {data_x.shape}")

    x_filled = np.vstack((imputed_tr, imputed_te))
    
    end = time()
    print(f"==== Dataset: {dataset} =====")
    print(f"Time taken: {end - start} seconds")
    mse = RMSE(x_filled, data_x, mask)
    mae = MAE(x_filled, data_x, mask)
    print(f"Missing Mechanism: {mode}, miss_rate: {p_miss}, RMSE: {mse}, MAE: {mae}")
    return x_filled, mse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="wine")
    parser.add_argument("--missing_mechanism", type=str, default="MCAR")
    parser.add_argument("--miss_rate", type=float, default=0.2)
    parser.add_argument("--rand_seed", type=int, default=1234)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    main(args.miss_rate, args.data_name, args.missing_mechanism, args.rand_seed)


