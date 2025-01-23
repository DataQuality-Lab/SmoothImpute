
import numpy as np
import argparse
import pandas as pd



import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './ReMasker_code')))
from remasker_impute import ReMasker


def remasker_imputation(xmiss):

    x_filled = xmiss.copy()
    x_filled = pd.DataFrame(x_filled)
    imputer = ReMasker()
    x_filled = imputer.fit_transform(x_filled)
    
    return pd.DataFrame(x_filled)


