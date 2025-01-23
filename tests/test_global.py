import numpy as np
from smoothimpute.imputers import Imputer
import pandas as pd

data = [[0, np.nan, 1],
        [np.nan, 1, 0]]

data_pd = pd.DataFrame(data)
imputer = Imputer("mice")
data_filled = imputer.impute(data_pd)
print(data_filled)
