# current implementation: only support numerical values
import numpy as np
import torch, os
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import argparse

class MaskEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, norm_layer=None):
        
        super().__init__()
        self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class ActiveEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, norm_layer=None):
        
        super().__init__()
        self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = torch.sin(x)
        x = x.transpose(1, 2)
        #   x = torch.cat((torch.sin(x), torch.cos(x + math.pi/2)), -1)
        x = self.norm(x)
        return x



def get_1d_sincos_pos_embed(embed_dim, pos, cls_token=False):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(pos)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def adjust_learning_rate(optimizer, epoch, lr, min_lr, max_epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        tmp_lr = lr * epoch / warmup_epochs 
    else:
        tmp_lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = tmp_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = tmp_lr
    return tmp_lr


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScaler:

    state_dict_key = "amp_scaler"
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



class MAEDataset(Dataset):

    def __init__(self, X, M):        
         self.X = X
         self.M = M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx]



def get_dataset(dataset : str, path : str):

    if dataset in ['climate', 'compression', 'wine', 'yacht', 'spam', 'letter', 'credit', 'raisin', 'bike', 'obesity', 'airfoil', 'blood', 'yeast', 'health', 'review', 'travel']:
        df = pd.read_csv(os.path.join(path, 'data', dataset + '.csv'))
        last_col = df.columns[-1]
        y = df[last_col]
        X = df.drop(columns=[last_col])
    elif dataset == 'california':
        from sklearn.datasets import fetch_california_housing
        X, y = fetch_california_housing(as_frame=True, return_X_y=True)
    elif dataset == 'diabetes':
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(as_frame=True, return_X_y=True)
    elif dataset == 'iris':
        # only for testing
        from sklearn.datasets import load_iris
        X, y = load_iris(as_frame=True, return_X_y=True)
    

    return X, y


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--dataset', default='california', type=str)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--max_epochs', default=600, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--embed_dim', default=32, type=int, help='embedding dimensions')
    parser.add_argument('--depth', default=6, type=int, help='encoder depth')
    parser.add_argument('--decoder_depth', default=4, type=int, help='decoder depth')
    parser.add_argument('--num_heads', default=4, type=int, help='number of heads')
    parser.add_argument('--mlp_ratio', default=4., type=float, help='mlp ratio')
    parser.add_argument('--encode_func', default='linear', type=str, help='encoding function')

    parser.add_argument('--norm_field_loss', default=False,
                        help='Use (per-patch) normalized field as targets for computing loss')
    parser.set_defaults(norm_field_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    ###### change this path
    parser.add_argument('--path', default='/data/tianyu/remasker/', type=str, help='dataset path')
    parser.add_argument('--exp_name', default='test', type=str, help='experiment name')

    # Dataset parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=666, type=int)

    parser.add_argument('--overwrite', default=True, help='whether to overwrite default config')
    parser.add_argument('--pin_mem', action='store_false')

    # distributed training parameters
    return parser

if __name__ == '__main__':
    
    X = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    X = X.unsqueeze(1)
    mask_embed = ActiveEmbed(4, 6)
    print(mask_embed(X).shape)


from typing import List
import argparse
import numpy as np

def parse_args() -> argparse.Namespace:
    """Generate args."""
    
    parser = argparse.ArgumentParser(description="data imputation with LLM")
    
    # System settings
    parser.add_argument("--data_name", type=str, help="The name of dataset to run.", default="restaurant")
    parser.add_argument("--output_dir", type=str, help="Output directory.", default="outputs")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_run", type=int, help="Number examples to run through model.", default=-1)
    parser.add_argument("--num_trials", type=int, help="Number trials to run. Results will be averaged with variance reported.", default=1)

    # Client settings
    parser.add_argument("--ckpt_dir", type=str, help="The path of the LLAMA model.", default="/data/jianweiw/LLM_Data_Quality/llama_finetune/models/llama-2-7b/")
    parser.add_argument("--tokenizer_path", type=str, help="The path of the tokenizer model.", default="/data/jianweiw/LLM_Data_Quality/llama_finetune/models/tokenizer.model")
    parser.add_argument("--max_seq_len", type=int, help="The size of the maximum sequence.", default=512)
    parser.add_argument("--max_gen_len", type=int, help="The size of the generated sequence.", default=256)
    parser.add_argument("--max_batch_size", type=int, help="The maximum of the batch size.", default=16)
    parser.add_argument("--temperature", type=float, help="The temperature to control the diversity.", default=0.6)
    parser.add_argument("--top_p", type=float, help="The threshold to control the probility.", default=0.9)

    # Prompt settings
    parser.add_argument("--task_instruction", type=str, help="The prompt to describe the task.", default=None)
    parser.add_argument("--k", type=int, help="Number examples in prompt", default=1)
    parser.add_argument("--sample_method", type=str, help="Example generation method", default="manual", 
                            choices=["random", "manual", "validation_clusters"])
    parser.add_argument("--sep_tok", type=str, help="Separate for attr: val pairs in row. Default is '.'.", default=".")
    parser.add_argument("--nan_tok", type=str, help="Token to represent nan entries. Default is 'nan'.",default="nan")

    # Imputation setting
    parser.add_argument("--miss_rate", type=float, help="The rate of the missing mechanism.", default=0.2)
    parser.add_argument('--missing_mechanism', choices=['MAR', 'MNAR','MCAR'], default='MCAR', type=str)
    
    args = parser.parse_args()
    
    return args


def compute_acc(preds: List, golds: List, data_m):
    """Compute accuracy."""
    mets = {"crc": 0, "total": 0}
    
    if data_m.shape[1] == 0:
        return 0
    for i in range(data_m.shape[0]):
        for j in range(data_m.shape[1]):
            if data_m[i][j] == 1:
                continue
            else:
                label = golds.iloc[i, j]
                pred = preds.iloc[i, j]
                label = label.strip().lower()
                pred = pred.strip().lower()
                label = label.replace("city", "").strip()
                pred = pred.replace("city", "").strip()
                mets["total"] += 1

                if label in pred:
                    mets["crc"] += 1

    acc = mets["crc"] / mets["total"]
    return acc

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


def update_imputation(
    data_table,
    preds,
    data_m,
    impute_cols: int = 2
):  
    num_rows, num_cols = data_table.shape
    headers = data_table.columns
    
    k = 0
    for i in range(num_rows):
        if data_m[i][impute_cols] == 1:
            continue
        data_table.iloc[i, impute_cols] = preds[k]
        k += 1
      
    return data_table
        

def normalization(data, parameters=None):
    # Parameters
    _, dim = data.shape
    norm_data = data.copy().astype(np.float64)
    
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

    
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy().astype(np.float64)
      
    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]
        
    return renorm_data


def normalization_pd(data, parameters=None):
    # Parameters
    _, dim = data.shape
    df_normalized = data.copy() 
    
    if parameters is None:
      
        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        index = 0
        # For each dimension
        for column in df_normalized.columns:
            min_val[index] = df_normalized[column].min()
            max_val[index] = df_normalized[column].max()
            df_normalized[column] = (df_normalized[column] - min_val[index]) / (max_val[index] - min_val[index])
            index+=1

        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                          'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
        
        # For each dimension
        index = 0
        for column in df_normalized.columns:

            df_normalized[column] = df_normalized[column]- min_val[index]
            df_normalized[column] = df_normalized[column] / (max_val[index] + 1e-6)  
            index+=1
        norm_parameters = parameters    
        
    return df_normalized, norm_parameters


def renormalization_pd (norm_data, norm_parameters):

    
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
      
    # for i in range(dim):
    index = 0
    for column in norm_data.columns:
        renorm_data[column] = renorm_data[column] * (max_val[index] + 1e-6)   
        renorm_data[column] = renorm_data[column] + min_val[index]
        index+=1
        
    return renorm_data


#### Accuracy Metrics ####
def MAE(X, X_true, mask):
    
    if mask.shape[1] == 0:
        return 0
    
    if mask.shape[1] == 1:
        X_true = X_true.reshape(-1, 1)
        X = X.reshape(-1, 1)
    
    X_true, norm_parameters = normalization(X_true)
    X, _ = normalization(X, norm_parameters)
    
    mask_ = ~mask.astype(bool)
    return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()



def RMSE(X, X_true, mask):
    
    if mask.shape[1] == 0:
        return 0
    
    if mask.shape[1] == 1:
        X_true = X_true.reshape(-1, 1)
        X = X.reshape(-1, 1)
    
    X_true, norm_parameters = normalization(X_true)
    X, _ = normalization(X, norm_parameters)
    mask_ = ~mask.astype(bool)
    # print(X_true[mask_])
    # print(X[mask_])
    return np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())



def compute_metrics(table_current, table, data_m, compute_index=None):
    
    column_types = table.dtypes
    numerical_index = []
    text_index = []
    acc, rmse, mae = 0, 0, 0
    for i in range(len(column_types)):
        if column_types[i] == "object":
            text_index.append(i)
        else:
            numerical_index.append(i)
    # calculate for all the table
    
    if compute_index is None:
    
        text_table_gt = table.iloc[:, text_index] 
        numerical_table_gt = table.iloc[:, numerical_index] 
        text_data_m = data_m[:, text_index]
        numerical_data_m = data_m[:, numerical_index]
        text_table_pred = table_current.iloc[:, text_index] 
        numerical_table_pred = table_current.iloc[:, numerical_index]
        
        acc = compute_acc(preds=text_table_pred, golds=text_table_gt, data_m=text_data_m)
        rmse = RMSE(X=numerical_table_pred.values, X_true=numerical_table_gt.values, mask=numerical_data_m)
        mae = MAE(X=numerical_table_pred.values, X_true=numerical_table_gt.values, mask=numerical_data_m)
    
    # calculate for one specifc column
    else:
        table_one_gt = table.iloc[:, compute_index]
        table_one_pred = table_current.iloc[:, compute_index]
        data_m_one = data_m[:, compute_index].reshape(-1, 1)
        if column_types[i] == "object":
            # acc = compute_acc(preds=table_one_pred, golds=table_one_gt , data_m=data_m_one)
            acc = compute_acc(preds=table_current, golds=table , data_m=data_m)
        else:
            rmse = RMSE(X=table_one_pred.values, X_true=table_one_gt.values, mask=data_m_one)
            mae = MAE(X=table_one_pred.values, X_true=table_one_gt.values, mask=data_m_one)
      

    return acc, rmse, mae

def rounding(table_current, table):

    column_types = table.dtypes

    for i in range(len(column_types)): 
        if column_types[i] == "int64":
            # table_current.iloc[:, i] = table_current.iloc[:, i].astype("float64")
            table_current.iloc[:, i] = table_current.iloc[:, i].apply(lambda x: float(int(round(float(x), 2))))
        else:
            table_current.iloc[:, i] = table_current.iloc[:, i].astype(column_types[i])
    
    return table_current
