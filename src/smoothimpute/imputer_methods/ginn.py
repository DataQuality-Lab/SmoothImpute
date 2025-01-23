import numpy as np

import sys
import os
  
import csv
import pandas as pd
from sklearn import model_selection, preprocessing

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './GINN_code')))
from ginn import GINN
from ginn.utils import degrade_dataset, data2onehot


def ginn_imputation(xmiss):

    mask = np.isnan(xmiss)
    y = np.zeros((xmiss.shape[0],1),dtype='int')
    
    cat_cols = []
    num_cols = [i for i in range(xmiss.shape[1])]
    X = xmiss.copy()
    y = np.reshape(y,-1)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, shuffle=False
    )
    
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

    x_filled = np.vstack((imputed_tr, imputed_te))
    
    return pd.DataFrame(x_filled)