
import numpy as np
import argparse
import pandas as pd



import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './GRAPE_code')))
from train_mdi import grape_impute


def grape_imputation(xmiss):

    x_filled = xmiss.copy()
    x_filled = pd.DataFrame(x_filled)
    x_filled = grape_impute(x_filled)
    
    return pd.DataFrame(x_filled)


