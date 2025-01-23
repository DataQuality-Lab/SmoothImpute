import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './TDM_code')))
from run_tdm import run_TDM
import pandas as pd

def tdm_imputation(xmiss, cuda_device, niter=10000, lr=1e-2, network_depth=3,  network_width=2, batch_size=64, report_interval=100):
    
    args = {'niter': niter, 'batchsize': batch_size, 'lr': lr, 'network_width': network_width, 'network_depth': network_depth, 'report_interval': report_interval}

    x_filled = run_TDM(xmiss, args)

    return pd.DataFrame(x_filled)