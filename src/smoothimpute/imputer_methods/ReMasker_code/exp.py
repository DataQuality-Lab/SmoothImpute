
import numpy as np
import argparse
import pandas as pd

from remasker_impute import ReMasker
import sys
import os
# 获取上一级目录的路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from utils_general import load_data_general
from utils_general import RMSE, MAE

from time import time

def main(p_miss=0.2, dataset="wine", mode="MCAR", rand_seed=1234):
    np.random.seed(rand_seed)

    start = time()
    data_x, xmiss, mask = load_data_general(data_name=dataset, miss_rate=p_miss, missing_mechanism=mode)
    

    x_filled = xmiss.copy()
    x_filled[mask == 0] = np.nan
    x_filled = pd.DataFrame(x_filled)
    imputer = ReMasker()
    x_filled = imputer.fit_transform(x_filled)
    
    end = time()
    print(f"==== Dataset: {dataset} ====")
    print(f"Time taken: {end - start} seconds")
    np.savetxt(f"../downstream_classification/ReMasker/{dataset}_filled.csv", x_filled, delimiter=",")
    np.savetxt(f"../downstream_classification/RAW/{dataset}_filled.csv", xmiss, delimiter=",")

    mse = RMSE(x_filled, data_x, mask)
    mae = MAE(x_filled, data_x, mask)
    print(f"Missing Mechanism: {mode}, miss_rate: {p_miss}, RMSE: {mse}, MAE: {mae}")
    return x_filled, mse

def arg_parse():
    parser = argparse.ArgumentParser('main settings', add_help=False)
    parser.add_argument("--data_name", type=str, default="wine")
    parser.add_argument("--missing_mechanism", type=str, default="MCAR")
    parser.add_argument("--miss_rate", type=float, default=0.2)
    parser.add_argument("--rand_seed", type=int, default=1234)
    return parser


if __name__ == "__main__":

    additional_args, unknown_args = arg_parse().parse_known_args()
    
    main(additional_args.miss_rate, additional_args.data_name, additional_args.missing_mechanism, additional_args.rand_seed)
    
    # import sys
    # data_name = sys.argv[1] if len(sys.argv) > 1 else "wine"
    # missing_mechanism = sys.argv[2] if len(sys.argv) > 2 else "MCAR"
    # miss_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    # rand_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1234
    # main(miss_rate, data_name, missing_mechanism, rand_seed)


