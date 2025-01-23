from typing import Any, Union
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
import time


class XGBoostImputation:
    def __init__(
        self,
        n_estimators: int = 100,
        max_iter: int = 10,
        random_state: Union[int, None] = None,
    ) -> None:
        """
        Initialize the XGBoost-based imputation class.

        Parameters:
        - n_estimators: int
            Number of trees in the XGBoost model.
        - max_iter: int
            Maximum number of iterations for iterative imputation.
        - random_state: Union[int, None]
            Random seed for reproducibility.
        """
        if not random_state:
            random_state = int(time.time())

        # Define the XGBoost regressor
        xgb_regressor = XGBRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            verbosity=0,
            objective="reg:squarederror"
        )
        
        # Define the iterative imputer
        self._model = IterativeImputer(
            estimator=xgb_regressor, random_state=random_state, max_iter=max_iter
        )

    def fit(self, X: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Fit the imputer on the given data and return the imputed values.

        Parameters:
        - X: np.ndarray
            Input data with missing values (np.nan).

        Returns:
        - np.ndarray
            Data with missing values imputed.
        """
        # Fit the imputer on the data
        self._model.fit(np.asarray(X), *args, **kwargs)
        
        # Transform the data to fill missing values
        imputed_X = self._model.transform(X)
        
        return imputed_X


import numpy as np
import argparse
import pandas as pd



def xgb_imputation(xmiss):

    x_filled = xmiss.copy()
    # x_filled = pd.DataFrame(x_filled)
    imputer = XGBoostImputation()
    x_filled = imputer.fit(x_filled)
    
    return pd.DataFrame(x_filled)


