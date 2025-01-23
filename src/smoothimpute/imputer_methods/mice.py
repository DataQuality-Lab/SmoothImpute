import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

def mice_imputation(xmiss):
    
    x_filled = IterativeImputer(random_state=0).fit_transform(xmiss)

    return pd.DataFrame(x_filled)