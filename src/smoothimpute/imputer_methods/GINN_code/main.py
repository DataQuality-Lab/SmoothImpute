import csv
import numpy as np
from sklearn import model_selection, preprocessing

from ginn import GINN
from ginn.utils import degrade_dataset, data2onehot
import warnings
warnings.filterwarnings("ignore")

datafile_w = './examples/heart.csv'

X = np.zeros((303,13),dtype='float')
y = np.zeros((303,1),dtype='int')
with open(datafile_w,'r') as f:
    reader=csv.reader(f)
    for i, row in enumerate(reader):
        data=[float(datum) for datum in row[:-1]]
        X[i]=data
        y[i]=row[-1]
        
cat_cols = [1,2,5,6,8,10,11,12,13]
num_cols = [0,3,4,7,9]
y = np.reshape(y,-1)
num_classes = len(np.unique(y))

missingness= 0.2
seed = 42

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, stratify=y
)
cx_train, cx_train_mask = degrade_dataset(x_train, missingness,seed, np.nan)
cx_test,  cx_test_mask  = degrade_dataset(x_test, missingness,seed, np.nan)

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

print(imputed_te)