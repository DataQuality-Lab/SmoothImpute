import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

from utils import normalization, renormalization, sample_batch_index, rounding, MAE, RMSE, uniform_sampler, binary_sampler

class Generator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Generator, self).__init__()

        self.Encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid()
        )

        self.Decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.Sigmoid()
        )

        self.fc_m = nn.Linear(hidden_channels, hidden_channels)
        self.fc_sigma = nn.Linear(hidden_channels, hidden_channels)
  
    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

# based on implementation of GAIN and material "https://adaning.github.io/posts/9047.html"
    def forward(self, x, m):
        inputs = torch.cat([x, m], axis = 1)
        
        code = self.Encoder(inputs)
        recon_x = self.Decoder(code)

        return recon_x
        
class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()

        self.w_1 = nn.Linear(in_channels, hidden_channels)
        self.w_2 = nn.Linear(hidden_channels, hidden_channels)
        self.w_3 = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, h):
        inputs = torch.cat([x, h], axis = 1)
        D_h1 = self.relu(self.w_1(inputs))
        D_h2 = self.relu(self.w_2(D_h1))
        Mask_prob = self.sigmoid(self.w_3(D_h2))

        return Mask_prob
    

def loss_computation(M, X, G_sample, Mask_prob, alpha):
    # mse loss
    loss_mse = torch.mean((M * X - M * G_sample)**2) / torch.mean(M)*1000

    # discriminator loss
    loss_discriminator = -torch.mean(M * torch.log(Mask_prob + 1e-8) + (1-M) * torch.log(1. - Mask_prob + 1e-8))

    return alpha*loss_mse  + loss_discriminator
    
    
def gain_imputation(xmiss, cuda_device, lr=0.0010, weight_decay=2e-4, epochs_num=10000, batch_size=64, hint_rate=0.9, alpha_mse=100):

        
    device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

    data_x = xmiss.copy()
    data_m = np.isnan(data_x)

    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    no, dim = data_x.shape
    
    h_dim = int(dim)  

    
    generator = Generator(2*dim, h_dim, dim).to(device)
    discriminator = Discriminator(2*dim, h_dim, dim).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=weight_decay)

    print('\n##### Start training...')


    for epoch in tqdm(range(epochs_num)):
        generator.train()
        optimizer_G.zero_grad()

        batch_idx = sample_batch_index(no, batch_size)

        X_mb = norm_data_x[batch_idx, :]  
        M_mb = data_m[batch_idx, :]  
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        G_sample = generator.forward(torch.Tensor(X_mb).to(device), torch.Tensor(M_mb).to(device))
        Hat_X = X_mb * M_mb + G_sample.cpu().detach().numpy() * (1-M_mb)
        Mask_prob = discriminator.forward(torch.Tensor(Hat_X).to(device), torch.Tensor(H_mb).to(device))
        
        loss = loss_computation(torch.Tensor(M_mb).to(device), torch.Tensor(X_mb).to(device), G_sample, Mask_prob, alpha_mse)
        
        loss.backward(retain_graph=True)
        optimizer_G.step()
        optimizer_D.step()

    

    start_test = time.time()
    generator.eval()
    a = [i for i in range(no)]
    batch_idx_ = np.array_split(a, batch_size)
    imputed_data_all = torch.Tensor([])
    for batch_idx in batch_idx_:
        X_mb = norm_data_x[batch_idx, :]  
        M_mb = data_m[batch_idx, :]  
        Z_mb = uniform_sampler(0, 0.01, len(batch_idx), dim)
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
    
        imputed_data= generator.forward(torch.Tensor(X_mb).to(device), torch.Tensor(M_mb).to(device))
        imputed_data = X_mb * M_mb + (1-M_mb) * imputed_data.cpu().detach().numpy()
        imputed_data_all = torch.cat((imputed_data_all, torch.Tensor(imputed_data)),dim=0)


    # Renormalization
    imputed_data = renormalization(imputed_data_all.cpu().detach().numpy(), norm_parameters)  
    
    # Rounding
    imputed_data = rounding(imputed_data, data_x) 

    return pd.DataFrame(imputed_data)

