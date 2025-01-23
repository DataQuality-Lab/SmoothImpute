import numpy as np
from sklearn.impute import KNNImputer
import pandas as pd

def knn_imputation(xmiss):
    
    imputer = KNNImputer(n_neighbors=5)
    x_filled = imputer.fit_transform(xmiss)

    return pd.DataFrame(x_filled)