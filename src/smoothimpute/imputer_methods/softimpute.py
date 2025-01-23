import numpy as np
from sklearn.utils.extmath import randomized_svd
import pandas as pd

class SoftImputeImputation:
    def __init__(self, shrinkage_value=10, max_iter=100, tol=1e-5):
        """
        Initialize the SoftImputeImputation class.

        Parameters:
        - shrinkage_value: float, the regularization parameter for soft-thresholding.
        - max_iter: int, maximum number of iterations.
        - tol: float, convergence tolerance for stopping criteria.
        """
        self.shrinkage_value = shrinkage_value
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(self, X):
        """
        Fit the SoftImpute model and fill in the missing values.

        Parameters:
        - X: ndarray, input matrix with missing values (NaNs).

        Returns:
        - X_filled: ndarray, input matrix with missing values filled.
        """
        # Create a mask for observed entries
        mask = ~np.isnan(X)

        # Replace missing values with zeros for initialization
        X_filled = np.nan_to_num(X.copy())

        for iteration in range(self.max_iter):
            # Perform SVD on the current filled matrix
            U, sigma, VT = randomized_svd(X_filled, n_components=min(X.shape), random_state=None)

            # Apply soft-thresholding to singular values
            sigma_thresholded = np.maximum(sigma - self.shrinkage_value, 0)

            # Reconstruct the matrix
            X_reconstructed = (U * sigma_thresholded) @ VT

            # Update only the missing entries
            X_new = X.copy()
            X_new[~mask] = X_reconstructed[~mask]

            # Check for convergence
            diff = np.linalg.norm(X_filled - X_new, ord='fro')
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Difference: {diff:.6f}")

            if diff < self.tol:
                print("Converged.")
                break

            X_filled = X_new

        return X_filled

def si_imputation(xmiss):

    # Create the imputer and fit_transform the data
    imputer = SoftImputeImputation(shrinkage_value=5, max_iter=500, tol=1e-5)
    x_filled = imputer.fit_transform(xmiss)

    return pd.DataFrame(x_filled)

