import numpy as np
import os
import torch
import logging
from tdm import TDM
import ot
from utils import MAE, RMSE
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import time


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')
data_dir = './datasets'

def normalization (data, parameters=None):
    
    _, dim = data.shape
    norm_data = data.copy()
    
    if parameters is None:
    
        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        
        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
            
        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                        'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
        
        # For each dimension
        for i in range(dim):
            norm_data[:,i] = norm_data[:,i] - min_val[i]
            norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
        
        norm_parameters = parameters    
        
    return norm_data, norm_parameters

def renormalization (norm_data, norm_parameters):
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data


def rounding(imputed_data, data_x):
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()
    
    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
        
    return rounded_data

def my_MAE(X, X_true, mask):
    # print(X[0])
    # print(X_true[0])
    # print(mask[0])
    X_true, norm_parameters = normalization(X_true)
    X, _ = normalization(X, norm_parameters)
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # should be an ndarray
        # mask_ = mask.astype(bool)
        mask_ = mask.astype(bool)
        # print(mask_[0])
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()


def my_RMSE(X, X_true, mask):
    
    X_true, norm_parameters = normalization(X_true)
    X, _ = normalization(X, norm_parameters)

    if torch.is_tensor(mask):
        # print("using torch")
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else: # should be an ndarray
        # print("using numpy")
        mask_ = mask.astype(bool)
        # mask_ = mask
        return np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())

def to_onehot(ori_data_x, data_x, data_m, category_idx):
    category_len_map = {}
    category_data_map = {}
    category_data_inv_map = {}
    new_data_x = []
    new_data_m = []
    for idx in category_idx:
        unique_data = np.unique(ori_data_x[:, idx])
        print(unique_data)
        data_map = {}
        data_inv_map = {}
        for i in range(unique_data.shape[0]):
            data_map[unique_data[i]] = i
            data_inv_map[i] = unique_data[i]
        category_data_map[idx] = data_map
        category_data_inv_map[idx] = data_inv_map
        category_len_map[idx] = unique_data.shape[0]
        temp_data_x = np.zeros((ori_data_x.shape[0], unique_data.shape[0]))
        temp_data_m = np.ones((ori_data_x.shape[0], unique_data.shape[0]))
        for i in range(ori_data_x.shape[1]):
            if data_m[i][idx] == 0:
                for j in range(unique_data.shape[0]):
                    temp_data_m[i][j] = 0
            else:
                temp_data_x[i][data_map[ori_data_x[i][idx]]] = 1
        new_data_x.append(temp_data_x)
        new_data_m.append(temp_data_m)

    new_data_x = np.concatenate((np.delete(data_x, category_idx, axis=1), np.concatenate(new_data_x, axis=1)), axis=1)
    new_data_m = np.concatenate((np.delete(data_m, category_idx, axis=1), np.concatenate(new_data_m, axis=1)), axis=1)

    return new_data_x, new_data_m, category_len_map, category_data_inv_map

def back_from_onehot(new_data_x, data_m, category_idx, ori_data_x, category_len_map, category_data_inv_map):
    not_category_idx = [i for i in range(ori_data_x.shape[1]) if i not in category_idx]
    # print(not_category_idx)

    new_predicted_data = []
    for idx in category_idx:
        category_len = category_len_map[idx]
        category_data = new_data_x[:, len(not_category_idx):len(not_category_idx)+category_len]
        new_data_x = np.delete(new_data_x, [i for i in range(len(not_category_idx), len(not_category_idx)+category_len)], axis=1)
        new_predicted_idx_data = ori_data_x[:, idx]
        for i in range(ori_data_x.shape[1]):
            if data_m[i][idx] == 0:
                predicted_data = np.argmax(category_data[i])
                new_predicted_idx_data[i] = category_data_inv_map[idx][predicted_data]
        new_predicted_data.append(new_predicted_idx_data)

    predicted_result = ori_data_x.copy()
    for i in range(len(category_idx)):
        predicted_result[:, category_idx[i]] = new_predicted_data[i].reshape(-1)
    for i in range(len(not_category_idx)):
        predicted_result[:, not_category_idx[i]] = new_data_x[:, i] 

    return predicted_result


def run_TDM(X_missing, args, X_true=None):
    # print(f"==== Dataset: {args['data_name']} ====")
    # print(f"==== Dataset: {args.data_name} ====")
    
    # For small datasets, smaller batchsize may prevent overfitting; 
    # For larger datasets, larger batchsize may give better performance.
    if 'batchsize' in args: 
        batchsize = args['batchsize']
    else:
        batchsize = 512
    

    X_missing = torch.Tensor(X_missing)
    if X_true is not None:
        X_true = torch.Tensor(X_true)
    n, d = X_missing.shape
    mask = torch.isnan(X_missing)

    k = args['network_width']
    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, k * d), nn.SELU(),  nn.Linear(k * d, k * d), nn.SELU(),
                            nn.Linear(k * d,  dims_out))
    projector = Ff.SequenceINN(d)
    for _ in range(args['network_depth']):
        projector.append(Fm.RNVPCouplingBlock, subnet_constructor=subnet_fc)

    imputer = TDM(projector,  batchsize=batchsize, im_lr=args['lr'], proj_lr=args['lr'], niter=args['niter'])
    imp = imputer.fit_transform(X_missing.clone(), verbose=True, report_interval=args['report_interval'], X_true=X_true)
    imp = imp.detach()

    result = {}
    result["imp"] = imp[mask.bool()].detach().cpu().numpy()
    
    imputed_data = rounding(imp.detach().cpu().numpy(), X_missing.detach().cpu().numpy()) 
    
    return imputed_data
