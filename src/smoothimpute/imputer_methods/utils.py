import torch.optim as optim
import numpy as np
import torch
from collections import Counter
from scipy import optimize
import pandas as pd


def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.
    
    Args:
        - total: total number of samples
        - batch_size: batch size
        
    Returns:
        - batch_idx: batch index
    '''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx

def uniform_sampler(low, high, rows, cols):
    '''Sample uniform random variables.
    
    Args:
        - low: low limit
        - high: high limit
        - rows: the number of rows
        - cols: the number of columns
        
    Returns:
        - uniform_random_matrix: generated uniform random matrix.
    '''
    return np.random.uniform(low, high, size = [rows, cols]) 

def binary_sampler(p, rows, cols):
    '''Sample binary random variables.
    
    Args:
        - p: probability of 1
        - rows: the number of rows
        - cols: the number of columns
        
    Returns:
        - binary_random_matrix: generated binary random matrix.
    '''

    unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
    binary_random_matrix = 1*(unif_random_matrix < p)
    return binary_random_matrix

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

def normalization(data, parameters=None):
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


def renormalization(norm_data, norm_parameters):
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


#### Accuracy Metrics ####
def MAE(X, X_true, mask):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.
    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.
    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)
    Returns
    -------
        MAE : float
    """
    X_true, norm_parameters = normalization(X_true)
    X, _ = normalization(X, norm_parameters)
    if torch.is_tensor(mask):
        # print("MAE using torch")
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # should be an ndarray
        # print("MAE using numpy")
        # mask_ = mask.astype(bool)
        mask_ = ~mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()



def RMSE(X, X_true, mask):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.
    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.
    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)
    Returns
    -------
        RMSE : float
    """
    X_true, norm_parameters = normalization(X_true)
    X, _ = normalization(X, norm_parameters)

    if torch.is_tensor(mask):
        # print("RMSE using torch")
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else: # should be an ndarray
        # print("RMSE using numpy")
        mask_ = ~mask.astype(bool)
        # mask_ = mask
        return np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())


