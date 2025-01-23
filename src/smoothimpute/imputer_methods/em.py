import numpy as np
import pandas as pd

class EMImputer:
    def __init__(self, max_iter=100, tol=1e-6):
        """
        初始化EM算法插补器。

        参数：
        - max_iter: 最大迭代次数。
        - tol: 收敛阈值。
        """
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(self, data):
        """
        使用EM算法对数据进行插补。

        参数：
        - data: 包含缺失值的数据矩阵，缺失值用np.nan表示。

        返回：
        - imputed_data: 插补后的数据矩阵。
        """
        data = np.array(data, dtype=np.float64)
        missing_mask = np.isnan(data)

        # 初始化缺失值为列均值
        col_means = np.nanmean(data, axis=0)
        data[missing_mask] = np.take(col_means, np.where(missing_mask)[1])

        for iteration in range(self.max_iter):
            old_data = data.copy()

            # E步：估计期望值
            mean = np.nanmean(data, axis=0)
            cov = np.cov(data, rowvar=False, bias=True)

            # 对每个缺失值进行估计
            for i in range(data.shape[0]):
                missing_indices = np.where(missing_mask[i])[0]
                observed_indices = np.where(~missing_mask[i])[0]

                if len(missing_indices) > 0:
                    observed_data = data[i, observed_indices]
                    
                    # 条件均值和条件协方差
                    sigma_oo = cov[np.ix_(observed_indices, observed_indices)]
                    sigma_om = cov[np.ix_(observed_indices, missing_indices)]
                    sigma_mo = cov[np.ix_(missing_indices, observed_indices)]
                    sigma_mm = cov[np.ix_(missing_indices, missing_indices)]

                    inv_sigma_oo = np.linalg.inv(sigma_oo)

                    conditional_mean = mean[missing_indices] + sigma_mo @ inv_sigma_oo @ (observed_data - mean[observed_indices])

                    # 插补缺失值
                    data[i, missing_indices] = conditional_mean

            # 检查收敛性
            if np.linalg.norm(data - old_data) < self.tol:
                print(f"EM算法在第{iteration + 1}次迭代时收敛。")
                break

        return data

def em_imputation(xmiss):

    # Create the imputer and fit_transform the data
    imputer = EMImputer(max_iter=500, tol=1e-6)
    x_filled = imputer.fit_transform(xmiss)

    return pd.DataFrame(x_filled)