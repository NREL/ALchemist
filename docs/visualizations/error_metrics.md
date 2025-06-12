# Error Metrics Visualization

The **Visualizations** dialog in ALchemist provides several tools to help you evaluate your surrogate model's performance using different error metrics. After training a model, you can access these visualizations to better understand how your model is performing as you add more experimental data.

---

## Available Error Metrics

You can select and plot the following metrics in the Visualizations dialog:

- **RMSE (Root Mean Squared Error):**  
  Measures the average magnitude of prediction errors. Lower RMSE indicates better fit. Sensitive to large errors.

$$
\mathrm{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$

- **MAE (Mean Absolute Error):**  
  The average of absolute differences between predicted and actual values. Less sensitive to outliers than RMSE.

$$
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

- **MAPE (Mean Absolute Percentage Error):**  
  Expresses prediction error as a percentage of the actual values. Useful for comparing errors across different scales, but can be unstable if actual values are near zero.

$$
\mathrm{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

- **$R^2$ (Coefficient of Determination):**  
  Indicates how well the model explains the variance in the data. Values closer to 1 mean better fit; values near 0 or negative indicate poor fit.

$$
R^2 = 1 - \frac{ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }{ \sum_{i=1}^{n} (y_i - \bar{y})^2 }
$$

Where:  
- $y_i$ = true value  
- $\hat{y}_i$ = predicted value  
- $\bar{y}$ = mean of true values  
- $n$ = number of data points

---

## Using the Visualizations Dialog

- **Plotting Metrics:**  
  Use the dropdown menu at the top of the Visualizations dialog to select which error metric to plot. Click "Plot Metrics" to see how the chosen metric changes as more data points are added.

- **Interpreting Trends:**  
  Ideally, error metrics (RMSE, MAE, MAPE) should decrease as you add more observations, and $R^2$ should increase. Flat or increasing error trends may indicate issues with model fit, data quality, or kernel choice.

- **Cross-Validation:**  
  All metrics are computed using cross-validation, providing a robust estimate of model performance on unseen data.

- **Parity Plots:**  
  In addition to error metrics, you can generate parity plots to visually compare predicted vs. actual values. Points close to the diagonal indicate good agreement.

---

## General Considerations

- Use error metric trends to monitor model improvement as you collect more data.
- Compare different metrics for a more complete picture of model performance.
- If you see unexpected trends, refer to the [Model Performance](../modeling/performance.md) page for troubleshooting tips and deeper guidance.

---

For more on interpreting these metrics and improving model performance, see the [Model Performance](../modeling/performance.md) section.