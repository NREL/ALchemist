# MIT License
#
# Copyright (c) 2022 Yannick Ureel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# 17 Feb 2025
# Adapted by Caleb Coatney from code originally written by Yannick Ureel.
# Modifications include switching from GPy to scikit-learn for Gaussian Process
# regression and adapting the acquisition function for the new framework.

import numpy as np
import pandas as pd
from skopt.space import Categorical
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import scipy
import scipy.special
import warnings


def gaussianAbsoluteMoment(muTilde, predVar, norm=1):
    """
    Compute the absolute moment of a Gaussian distribution.

    This function computes the expected absolute value of a Gaussian-distributed variable
    raised to the given norm.

    Parameters:
        muTilde (np.ndarray): Mean values of the Gaussian distribution.
        predVar (np.ndarray): Variance values of the Gaussian distribution.
        norm (int): The order of the absolute moment. Default is 1.

    Returns:
        np.ndarray: Computed absolute moments.
    """
    # Compute the confluent hypergeometric function
    f11 = scipy.special.hyp1f1(-0.5 * norm, 0.5, -0.5 * np.divide(muTilde ** 2, predVar))
    
    # Compute prefactors
    prefactors = ((2 * predVar ** 2) ** (norm / 2.0) * scipy.special.gamma((1 + norm) / 2.0)) / np.sqrt(np.pi)
    
    return np.multiply(prefactors, f11)


def calcEMOC(x, X, kernel, model, sigmaN, norm=1):
    """
    Compute the Expected Model Output Change (EMOC) scores.

    EMOC is a metric used for active learning to quantify the expected impact of selecting
    a candidate point on the model output.

    Parameters:
        x (np.ndarray): Candidate points to evaluate EMOC scores.
        X (np.ndarray): Observed data points.
        kernel (Callable): Kernel function from scikit-learn or scikit-optimize.
        model (GaussianProcessRegressor): Trained Gaussian Process model.
        sigmaN (float): Noise variance.
        norm (int): Norm for absolute moment calculation. Default is 1.

    Returns:
        np.ndarray: EMOC scores for the candidate points.
    """
    emocScores = np.empty((x.shape[0], 1), dtype=np.float64)

    # Compute the full kernel matrix for observed (X) and candidate (x) points
    kAll = kernel(np.vstack([X, x]))  # This concatenates X and x to compute the full covariance matrix.
    
    # Extract the covariance between observed (X) and candidate (x) points
    k = kAll[:X.shape[0], X.shape[0]:]

    # Extract the diagonal values of the candidate-candidate covariance matrix
    # This is the self-covariance (variance) for each candidate point
    selfKdiag = np.diag(kAll[X.shape[0]:, X.shape[0]:]).reshape(-1, 1)

    # Get GP model predictions for the candidate points (mean and standard deviation)
    muTilde, sigmaF = model.predict(x, return_std=True)
    sigmaF = sigmaF.reshape(-1, 1) ** 2  # Convert standard deviation to variance

    # Compute Gaussian absolute moment correction for model predictions
    moments = gaussianAbsoluteMoment(muTilde, sigmaF, norm)

    # Compute the first term in the EMOC equation
    term1 = 1.0 / (sigmaF + sigmaN)

    # Compute the second term by solving a linear system (K_xx + sigmaN * I) * term2 = k
    K_xx = kernel(X, X) + np.eye(X.shape[0]) * sigmaN
    term2 = np.full((X.shape[0] + 1, x.shape[0]), -1.0, dtype=np.float64)
    term2[:X.shape[0], :] = np.linalg.solve(K_xx, k)

    # Precompute multiplication for efficiency in later steps
    preCalcMult = np.dot(term2[:-1, :].T, kAll[:X.shape[0], :])  # Shape (x.shape[0], X.shape[0])

    # Compute EMOC scores for each candidate point
    for idx in range(x.shape[0]):
        # Perform calculations based on the terms computed earlier
        vAll = term1[idx] * (
            preCalcMult[idx, :X.shape[0]] + np.dot(term2[-1, idx], kAll[X.shape[0] + idx, :X.shape[0]])
        )
        # The EMOC score is the mean of the absolute values raised to the norm power
        emocScores[idx] = np.mean(np.abs(vAll) ** norm)

    # The final EMOC scores are corrected with the absolute moment adjustment
    # Using np.diag ensures that the final EMOC scores are returned as a 1D array
    # (flattening the result) after applying the moment correction.
    return np.diag(np.multiply(emocScores, moments))  # The diag ensures it's a 1D vector


def select_EMOC(pool, X, y, search_space, model=None, verbose=False):
    """Select a point using Expected Maximal Output Change (EMOC)."""
    # Check if model is a BaseModel subclass
    if hasattr(model, 'predict'):
        # Determine the point in the pool that would maximize output change
        if verbose:
            print("Selecting point that maximizes expected output change...")
        
        predictions = []
        stds = []
        
        for i, point in enumerate(pool.iterrows()):
            point_df = pd.DataFrame([point[1]])
            # Use the model's predict method
            pred, std = model.predict(point_df, return_std=True)
            predictions.append(pred[0])
            stds.append(std[0])
            
        # Find the point with the maximum expected change
        max_idx = np.argmax(stds)
        
        # Return the selected point as a DataFrame
        return pd.DataFrame([pool.iloc[max_idx]])
    else:
        # Fall back to original implementation for compatibility
        # Identify categorical columns from the search space.
        cat_cols = [dim.name for dim in search_space if isinstance(dim, Categorical)]

        # One-hot encode categorical columns, dropping one dummy per category.
        if cat_cols:
            # One-hot encode for pool and X, dropping the first category for each categorical variable
            encoded_pool = pd.get_dummies(pool, columns=cat_cols, drop_first=True).astype(float)
            encoded_X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)

            # Get non-categorical columns (they remain unchanged).
            non_cat_cols = [col for col in X.columns if col not in cat_cols]

            # Build full list of dummy columns for each categorical variable (excluding dropped ones).
            full_dummy_columns = non_cat_cols.copy()
            for dim in search_space:
                if dim.name in cat_cols and hasattr(dim, 'categories'):
                    for cat_val in dim.categories[1:]:  # Skip first category to match drop_first=True
                        full_dummy_columns.append(f"{dim.name}_{cat_val}")

            # Reindex both dataframes to ensure they have the correct set of columns.
            encoded_pool = encoded_pool.reindex(columns=full_dummy_columns, fill_value=0).astype(float)
            encoded_X = encoded_X.reindex(columns=full_dummy_columns, fill_value=0).astype(float)

        else:
            encoded_pool = pool.copy().astype(float)
            encoded_X = X.copy().astype(float)

        # If no model is provided, create and notify the user of the default model.
        if model is None:
            # Initialize lengthscales using the mean of each column in the encoded observed data.
            ls_init = np.mean(encoded_X.values, axis=0)
            default_kernel = C() * RBF(length_scale=ls_init)
            print("Notice: Using default GaussianProcessRegressor model with parameters:")
            print("  Kernel:", default_kernel)
            print("  Optimizer: fmin_l_bfgs_b")
            print("  n_restarts_optimizer: 20")
            print("  random_state: 42")
            model = GaussianProcessRegressor(kernel=default_kernel,
                                             optimizer='fmin_l_bfgs_b',
                                             n_restarts_optimizer=20,
                                             random_state=42)
        else:
            # Check that the provided model inherits from GaussianProcessRegressor.
            if not isinstance(model, GaussianProcessRegressor):
                raise ValueError("The provided model must be an instance of or inherit from "
                                 "sklearn.gaussian_process.GaussianProcessRegressor.")

        # Fit the Gaussian Process model on the encoded observed data.
        model.fit(encoded_X.values, y.values)
        # Use the updated kernel from the fitted model.
        kernel = model.kernel_

        # Extract the learned lengthscale vector and noise variance.
        lengthscales = kernel.get_params()["k2__length_scale"]
        if verbose:
            print(f"Learned Lengthscales: {lengthscales}")
        sigmaN = model.alpha_
        if verbose:
            print(f"Noise variance (sigmaN): {sigmaN}")

        # Compute EMOC scores on the encoded pool using the encoded observed data.
        var = calcEMOC(encoded_pool.values, encoded_X.values, kernel, model, sigmaN)

        # Ensure var is a 1D array; if not, extract the diagonal.
        if var.ndim == 2:
            warnings.warn(f"Warning: var is {var.shape}, taking diagonal")
            var = np.diag(var)
        elif var.shape[0] != pool.shape[0]:
            raise ValueError(f"var has {var.shape[0]} elements, but pool has {pool.shape[0]}")

        # Identify the candidate point with the maximum EMOC score.
        best_index = np.argmax(var)

        # Double-check that the shape is correct (should match the original pool size).
        if var.shape[0] != pool.shape[0]:
            raise ValueError(f"Mismatch: var has {var.shape[0]} elements, but pool has {pool.shape[0]}")

        # Return the best candidate point from the original pool with original feature names.
        best_point = pool.iloc[best_index]
        return pd.DataFrame([best_point], columns=pool.columns)


