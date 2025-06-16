# scikit-optimize Backend

The **scikit-optimize** backend in ALchemist allows you to train a Gaussian Process (GP) surrogate model using the `skopt.learning.GaussianProcessRegressor`, which is a wrapper around `sklearn.gaussian_process.GaussianProcessRegressor` designed specifically for Bayesian optimization workflows.

---

## What is scikit-optimize?

[scikit-optimize (skopt)](https://scikit-optimize.github.io/) is a library built on top of scikit-learn that provides tools for sequential model-based optimization, commonly used in Bayesian optimization. In ALchemist, the skopt backend leverages this framework to efficiently model your experimental data and suggest new experiments.

---

## Training a Model with scikit-learn Backend

When you select the **scikit-learn** backend in the Model panel, you are training a Gaussian Process model using the skopt/scikit-learn stack. The workflow and options are as follows:

### 1. Kernel Selection

You can choose from several kernel types for the GP:

- **RBF (Radial Basis Function):** Default, smooth kernel.
- **Matern:** Flexible kernel with a tunable smoothness parameter (`nu`).
- **RationalQuadratic:** Mixture of RBF kernels with varying length scales.

For the Matern kernel, you can select the `nu` parameter (0.5, 1.5, 2.5, or ∞), which controls the smoothness of the function.

> **Note:** ALchemist uses anisotropic (dimension-wise) kernels by default, so each variable can have its own learned lengthscale. This helps preserve the physical meaning of each variable and enables automatic relevance detection (ARD). For more details, see the [Kernel Deep Dive](../background/kernels.md) in the Educational Resources section.

### 2. Optimizer

You can select the optimizer used for hyperparameter tuning:

- **L-BFGS-B** (default)
- **CG**
- **BFGS**
- **TNC**

These control how the kernel hyperparameters are optimized during model fitting.

### 3. Advanced Options

Enable advanced options to customize kernel and optimizer settings. By default, kernel hyperparameters are automatically optimized.

### 4. Noise Handling

If your experimental data includes a `Noise` column, these values are used for regularization (`alpha` parameter in scikit-learn). If not, the model uses its default regularization.

### 5. One-Hot Encoding

Categorical variables are automatically one-hot encoded for compatibility with scikit-learn.

### 6. Model Training and Evaluation

- The model is trained on your current experiment data.
- Cross-validation is performed to estimate model performance (RMSE, MAE, MAPE, R²).
- Learned kernel hyperparameters are displayed after training.

---

## How It Works

- The model uses your variable space and experiment data to fit a GP regression model.
- The trained model is used for Bayesian optimization, suggesting new experiments via acquisition functions.
- All preprocessing (encoding, noise handling) is handled automatically.

---

## References

- [scikit-learn GaussianProcessRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
- [scikit-optimize GaussianProcessRegressor documentation](https://scikit-optimize.github.io/stable/modules/generated/skopt.learning.GaussianProcessRegressor.html)

---

For a deeper explanation of kernel selection, anisotropic kernels, and ARD, see [Kernel Deep Dive](../background/kernels.md) in the Educational Resources section.

For details on using the BoTorch backend, see the next section.