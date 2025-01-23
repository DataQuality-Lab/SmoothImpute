import argparse
import time
import random
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from neural_tangents import stax
import neural_tangents as nt
import jax
import hnswlib
import pandas as pd

def dist2sim(neigh_dist):
    if torch.is_tensor(neigh_dist):
        neigh_dist = neigh_dist.cpu().detach().numpy()
    with np.errstate(divide="ignore"):
        dist = 1.0 / neigh_dist
    inf_mask = np.isinf(dist)
    inf_row = np.any(inf_mask, axis=1)
    dist[inf_row] = inf_mask[inf_row]
    denom = np.sum(dist, axis=1)
    denom = denom.reshape((-1,1))
    return dist/denom

def normalization (data, parameters=None):
    '''Normalize data in [0, 1] range.
    
    Args:
        - data: original data
    
    Returns:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each feature for renormalization
    '''

    # Parameters
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
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data


def rounding(imputed_data, data_x):
    '''Round imputed data for categorical variables.
    
    Args:
        - imputed_data: imputed data
        - data_x: original data with missing values
        
    Returns:
        - rounded_data: rounded imputed data
    '''
    
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()
    
    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
        
    return rounded_data

def prediction(pred_fn, X_test, kernel_type="nngp", compute_cov = True):

		pred_mean, pred_cov = pred_fn(x_test=X_test, get=kernel_type, compute_cov= compute_cov)
		return pred_mean, pred_cov

def normalization_std(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data+1

def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.
    
    Args:
        - total: total number of samples
        - batch_size: batch size
        
    Returns:
        - batch_idx: batch index
    '''
    total_idx = np.random.permutation(total)
    #   total_idx = np.arange(0, total)
    batch_idx = total_idx[:batch_size]

    return batch_idx
    

def nomi_imputation(xmiss):

    norm_data, norm_parameters = normalization(xmiss)
    data_m = ~np.isnan(xmiss)

    norm_data_x = np.nan_to_num(norm_data, 0)

    num, dims = norm_data_x.shape
    imputed_X = norm_data_x.copy()
    data_m_imputed = data_m.copy()

    init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(2*dims), stax.Relu(),
    stax.Dense(dims), stax.Relu(),
    stax.Dense(1), stax.Sigmoid_like()
    )
    max_iter = 3

    if num < 50000:
        for iteration in range(max_iter):
            for dim in tqdm(range(dims)):
                
                X_wo_dim = np.delete(imputed_X, dim, 1)
                i_not_nan_index = data_m_imputed[:, dim].astype(bool)
                
                X_train = X_wo_dim[i_not_nan_index]
                Y_train = imputed_X[i_not_nan_index, dim]

                X_test = X_wo_dim[~i_not_nan_index]
                true_indices = np.where(~i_not_nan_index)[0]
                # print(~i_not_nan_index, true_indices)
                
                if X_test.shape[0] == 0:
                    continue

                no, d = X_train.shape
                
                index = hnswlib.Index(space="l2", dim=d)
                index.init_index(max_elements=no, ef_construction=200, M=16)
                index.add_items(X_train)
                index.set_ef(int(12))

                if X_train.shape[0]>300:
                    batch_idx = sample_batch_index(X_train.shape[0], 300)
                else:
                    batch_idx = sample_batch_index(X_train.shape[0], X_train.shape[0])
                
                X_batch = X_train[batch_idx,:]
                Y_batch = Y_train[batch_idx]

                neigh_ind, neigh_dist = index.knn_query(X_batch, k=10)
                neigh_dist = np.sqrt(neigh_dist)

                weights = dist2sim(neigh_dist[:,1:])
                
                y_neighbors = Y_train[neigh_ind[:,1:]]
                train_input = weights*y_neighbors
                
                neigh_ind_test, neigh_dist_test = index.knn_query(X_test, k=10)
                neigh_dist_test = np.sqrt(neigh_dist_test)

                weights_test = dist2sim(neigh_dist_test[:, :-1])
                y_neighbors_test = Y_train[neigh_ind_test[:, :-1]]
                test_input = weights_test*y_neighbors_test

                predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, 
                        train_input, Y_batch.reshape(-1, 1), diag_reg=1e-4)
                
                y_pred, pred_cov = prediction(predict_fn, test_input, kernel_type="nngp")
                

                if iteration == 0:
                    imputed_X[~i_not_nan_index, dim] = y_pred.reshape(-1)
                elif iteration <= 3:
                    pred_std = np.sqrt(np.diag(pred_cov))
                    pred_std = np.ravel(np.array(pred_std))
                    pred_std = normalization_std(pred_std)
                    
                    pred_std = np.nan_to_num(pred_std, nan=1.0)
                    
                    greater_than_threshold_indices = np.where(pred_std <= 1.0)[0]
                    
                    for i in range(greater_than_threshold_indices.shape[0]):
                        data_m_imputed[true_indices[greater_than_threshold_indices[i]]:, dim] = 1
                    
                    imputed_X[~i_not_nan_index, dim] = (1-1.0/pred_std)*imputed_X[~i_not_nan_index, dim] + 1.0/pred_std*y_pred.reshape(-1)
                else:
                    imputed_X[~i_not_nan_index, dim] = y_pred.reshape(-1)

        imputed_data = renormalization(imputed_X, norm_parameters)  
        imputed_data = rounding(imputed_data, xmiss)
    else:
        for iteration in range(max_iter):
            
            all_train_input = []
            all_test_input = []
            all_train_label = []

            for dim in range(dims):

                X_wo_dim = np.delete(imputed_X, dim, 1)
                i_not_nan_index = data_m_imputed[:, dim].astype(bool)
                X_train = X_wo_dim[i_not_nan_index] 
                Y_train = imputed_X[i_not_nan_index, dim]

                X_test = X_wo_dim[~i_not_nan_index]
                
                no, d = X_train.shape

                index = hnswlib.Index(space="l2", dim=d)
                index.init_index(max_elements=no, ef_construction=200, M=16)
                index.add_items(X_train)
                index.set_ef(int(12))

                if X_train.shape[0]>50:
                    batch_idx = sample_batch_index(X_train.shape[0], 50)
                else:
                    batch_idx = sample_batch_index(X_train.shape[0], X_train.shape[0])

                X_batch = X_train[batch_idx,:]
                Y_batch = Y_train[batch_idx]

                neigh_ind, neigh_dist = index.knn_query(X_batch, k=10)
                neigh_dist = np.sqrt(neigh_dist)

                weights = dist2sim(neigh_dist[:,1:])
                
                y_neighbors = Y_train[neigh_ind[:,1:]]
                train_input = weights*y_neighbors
                
                neigh_ind_test, neigh_dist_test = index.knn_query(X_test, k=10)
                neigh_dist_test = np.sqrt(neigh_dist_test)

                weights_test = dist2sim(neigh_dist_test[:, :-1])
                y_neighbors_test = Y_train[neigh_ind_test[:, :-1]]
                test_input = weights_test*y_neighbors_test

                all_train_input.append(train_input)
                all_train_label.append(Y_batch.reshape(-1, 1))
                all_test_input.append(test_input)
            
            all_train_input = np.vstack(all_train_input)
            all_train_label = np.vstack(all_train_label)
            all_test_input = np.vstack(all_test_input)

            split_size = 10
            no_test, dim_test = all_test_input.shape
            additional_sample_num = (int(no_test/split_size)+1)*split_size-no_test
            additional_sample = np.zeros((additional_sample_num, dim_test))
            all_test_input = np.vstack((all_test_input, additional_sample))

            all_train_input = all_train_input.reshape(-1, 9*split_size)
            all_train_label = all_train_label.reshape(-1, split_size)
            all_test_input = all_test_input.reshape(-1, 9*split_size)

            

            print(all_train_input.shape, all_train_label.shape, all_test_input.shape)
            print("start nngp training")

            predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, 
                    all_train_input, all_train_label, diag_reg=1e-4)

            predict_batch_size = 2048
            y_pred = np.zeros((all_test_input.shape[0], split_size))
            for i in tqdm(range(0, all_test_input.shape[0], predict_batch_size)):
                    # print(iteration, i)
                    index_end = min(i+predict_batch_size, all_test_input.shape[0])
                    # y_pred[i:index_end] = prediction(predict_fn, test_input[i:index_end], kernel_type="nngp")
                    y_pred_batch, pred_cov = prediction(predict_fn, all_test_input[i:index_end], kernel_type="nngp")
                    y_pred[i:index_end] = y_pred_batch
            
            y_pred = y_pred.reshape(-1, 1)
            y_pred = y_pred[0:-1*additional_sample_num,:]

            count = 0
            for dim in range(dims):
                i_not_nan_index = data_m_imputed[:, dim].astype(bool)
                imputed_X[~i_not_nan_index, dim] = y_pred[count:count+np.count_nonzero(~i_not_nan_index),:].reshape(-1)
                count += np.count_nonzero(~i_not_nan_index)
            
        imputed_data = renormalization(imputed_X, norm_parameters)  
        imputed_data = rounding(imputed_data, xmiss)

    return pd.DataFrame(imputed_data)