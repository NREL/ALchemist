# BoTorch Backend

The **BoTorch** backend in ALchemist allows you to train a Gaussian Process (GP) surrogate model using the [BoTorch](https://botorch.org/) library, which is built on PyTorch and designed for scalable Bayesian optimization. BoTorch provides advanced kernel options and efficient handling of both continuous and categorical variables.

---

## What is BoTorch?

[BoTorch](https://botorch.org/) is a flexible, research-oriented library for Bayesian optimization built on PyTorch. It was developed by Meta (Facebook) and serves as the underlying optimization engine for [Ax](https://ax.dev/), a high-level adaptive experimentation platform. BoTorch supports advanced features such as anisotropic kernels (automatic relevance determination, ARD) and mixed-variable spaces, and is tightly integrated with [GPyTorch](https://gpytorch.ai/) for scalable Gaussian process inference.

---

## Training a Model with BoTorch Backend

When you select the **botorch** backend in the Model panel, you are training a GP model using BoTorch's `SingleTaskGP` (for continuous variables) or `MixedSingleTaskGP` (for mixed continuous/categorical variables). The workflow and options are as follows:

### 1. Kernel Selection

You can choose the kernel type for the continuous variables:

- **Matern:** Default, with a tunable smoothness parameter (`nu`).

- **RBF:** Radial Basis Function kernel.

For the Matern kernel, you can select the `nu` parameter (0.5, 1.5, or 2.5), which controls the smoothness of the function.

> **Note:** BoTorch uses anisotropic (ARD) kernels by default, so each variable can have its own learned lengthscale. This helps preserve the physical meaning of each variable and enables automatic relevance detection. For more details, see the [Kernel Deep Dive](../background/kernels.md) in the Educational Resources section.

### 2. Categorical Variables

- Categorical variables are automatically detected and encoded.

- BoTorch uses the `MixedSingleTaskGP` model to handle mixed spaces, encoding categorical variables as required.

### 3. Noise Handling

- If your experimental data includes a `Noise` column, these values are used for regularization.

- If not, the model uses its internal noise estimation.

### 4. Model Training and Evaluation

- The model is trained on your current experiment data.

- Cross-validation is performed to estimate model performance (RMSE, MAE, MAPE, R²).

- Learned kernel hyperparameters (lengthscales, outputscale, etc.) are displayed after training.

### 5. Input and Output Transforms

**Smart Defaults (v0.3.0+):**

BoTorch models in ALchemist now automatically apply input normalization and output standardization for improved performance:

- **Input Normalization**: Scales inputs to [0, 1] range, improving numerical stability

- **Output Standardization**: Centers outputs to zero mean and unit variance, helping with optimization

These transforms are applied automatically unless explicitly overridden. This typically improves model R² from ~0.0001 to 0.3-0.9 for typical problems.

**Manual Override:**

You can explicitly specify transform types when needed:

- `input_transform_type`: "normalize" (default), "standardize", or "none"

- `output_transform_type`: "standardize" (default) or "none"

### 6. Advanced Options

- You can select the kernel type and Matern `nu` parameter in the Model panel.

- BoTorch uses sensible defaults for training iterations and transforms.

- ARD lengthscales are extracted and displayed after training for feature importance analysis.

---

## How It Works

- The model uses your variable space and experiment data to fit a GP regression model using PyTorch.

- Input/output transforms are automatically applied for better performance.

- The trained model is used for Bayesian optimization, suggesting new experiments via acquisition functions.

- All preprocessing (encoding, transforms, noise handling) is handled automatically.

---

## References

- [BoTorch documentation](https://botorch.org/docs/introduction)
- [BoTorch SingleTaskGP](https://botorch.readthedocs.io/en/latest/models.html#botorch.models.gp_regression.SingleTaskGP)
- [BoTorch MixedSingleTaskGP](https://botorch.readthedocs.io/en/latest/models.html#botorch.models.gp_regression_mixed.MixedSingleTaskGP)
- [Ax platform documentation](https://ax.dev/)
- [GPyTorch documentation](https://gpytorch.ai/)

---

For a deeper explanation of kernel selection, anisotropic kernels, and ARD, see [Kernel Deep Dive](../background/kernels.md) in the Educational Resources section.

For details on using the scikit-optimize backend, see the previous section.