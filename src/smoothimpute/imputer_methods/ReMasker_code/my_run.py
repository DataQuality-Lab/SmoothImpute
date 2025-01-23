import numpy as np
import pandas as pd
from utils import get_args_parser, binary_sampler, compute_metrics
from remasker_impute import ReMasker

X_raw = np.arange(50).reshape(10, 5) * 1.0
X = pd.DataFrame(X_raw, columns=['0', '1', '2', '3', '4'])
X.iat[3,0] = np.nan

data_path =  "../data_update/wine.csv"
table = pd.read_csv(data_path, index_col=None)
num_rows, num_cols = table.shape
data_m = binary_sampler(1-0.2, num_rows, num_cols)

table_current = table.copy()

table_current[data_m == 0] = np.nan


imputer = ReMasker()

imputed = imputer.fit_transform(table_current)

print(table.head())
print(table_current.head())
print(imputed)

acc, rmse, mae = compute_metrics(table_current, table, data_m, compute_index=None)
print(f"The acc is {acc}. The imputation rmse is {rmse}. The imputation mae is {mae}.")

acc, rmse, mae = compute_metrics(pd.DataFrame(imputed), table, data_m, compute_index=None)
print(f"The acc is {acc}. The imputation rmse is {rmse}. The imputation mae is {mae}.")
# print(imputed[3,0])