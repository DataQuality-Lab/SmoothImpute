import numpy as np
import pandas as pd

def mean_imputation(xmiss):
    
    x_filled = xmiss.copy()
    nan_mask = np.isnan(xmiss)
    mean_values = np.nanmean(xmiss, axis=0)
    x_filled[nan_mask] = np.take(mean_values, np.where(nan_mask)[1])

    return pd.DataFrame(x_filled)