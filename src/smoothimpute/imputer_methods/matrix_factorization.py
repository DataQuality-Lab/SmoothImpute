
import numpy as np
import argparse
import pandas as pd

import numpy as np

class MatrixFactorizationImputation:
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.1, max_iter=1000, tol=1e-4):
        """
        Initialize the MatrixFactorizationImputation class.

        Parameters:
        - n_factors: int, number of latent factors.
        - learning_rate: float, learning rate for gradient descent.
        - regularization: float, regularization term for L2 penalty.
        - max_iter: int, maximum number of iterations.
        - tol: float, convergence tolerance for stopping criteria.
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(self, X):
        """
        Fit the matrix factorization model and fill in the missing values.

        Parameters:
        - X: ndarray, input matrix with missing values (NaNs).

        Returns:
        - X_filled: ndarray, input matrix with missing values filled.
        """
        # Create a mask for observed entries
        mask = ~np.isnan(X)

        # Initialize user and item latent factor matrices
        m, n = X.shape
        P = np.random.normal(scale=0.1, size=(m, self.n_factors))
        Q = np.random.normal(scale=0.1, size=(n, self.n_factors))

        # Replace missing values with zeros for computation
        X_filled = np.nan_to_num(X)

        for iteration in range(self.max_iter):
            # Compute predictions
            X_pred = P @ Q.T

            # Compute gradient only for observed entries
            error = (X_filled - X_pred) * mask

            # Compute gradients
            P_grad = -2 * (error @ Q) + 2 * self.regularization * P
            Q_grad = -2 * (error.T @ P) + 2 * self.regularization * Q

            # Update latent factors
            P -= self.learning_rate * P_grad
            Q -= self.learning_rate * Q_grad

            # Calculate loss (only on observed entries)
            loss = np.sum((error ** 2)[mask]) + self.regularization * (np.sum(P**2) + np.sum(Q**2))

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")

            # Check convergence
            if loss < self.tol:
                print("Converged.")
                break

        # Fill in missing values with predictions
        X_filled[~mask] = (P @ Q.T)[~mask]

        return X_filled




def mf_imputation(xmiss):

    x_filled = xmiss.copy()
    imputer = MatrixFactorizationImputation()
    x_filled = imputer.fit_transform(x_filled)
    
    return pd.DataFrame(x_filled)