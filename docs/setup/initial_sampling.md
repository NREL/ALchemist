# Generating Initial Experiments

When starting an active learning workflow, it's important to generate an initial set of experiments that cover your variable space efficiently. This is especially useful if you have no prior experimental data, or if your existing data was not collected with surrogate modeling in mind. Well-chosen initial points help ensure that your model converges efficiently and avoids bias from poor coverage.

---

## Why Generate Initial Points?

- **No Prior Data:** If you are starting from scratch, you need a set of initial experiments to train your first surrogate model.

- **Supplement Existing Data:** If you have some data, but it is sparse or not well-distributed, you can generate additional points to improve coverage.

- **Efficient Model Convergence:** Good initial coverage of the variable space helps the model learn faster and reduces the risk of missing important regions.

---

## How to Generate Initial Points

1. **Load or Define Your Variable Space:**  
   Make sure you have set up your variable space using the Variable Space Setup dialog.

2. **Open the Initial Sampling Dialog:**  
   In the main application window, click **Generate Initial Points** in the Experiment Data panel.

3. **Choose a Sampling Strategy:**  
   Select from several strategies:
   - **Random:** Uniformly samples points at random.
   - **LHS (Latin Hypercube Sampling):** Ensures each variable is sampled evenly across its range.
   - **Sobol, Halton, Hammersly:** Quasi-random low-discrepancy sequences for more uniform coverage in high dimensions.

4. **Set the Number of Points:**  
   Enter how many initial experiments you want to generate.

5. **Generate and Review:**  
   Click **Generate**. The new points will appear in your experiment table, ready for export or further editing.

---

## Tips

- **Coverage Matters:** More points give better coverage, but also require more experiments. Balance your resources and modeling needs.

- **Quasi-Random vs. Random:** Quasi-random methods (LHS, Sobol, etc.) are generally preferred for initial sampling, especially in higher dimensions.

- **Supplementing Data:** You can generate initial points even if you already have some data, to fill gaps or improve distribution.

---

## Example Workflow

1. Define your variable space (e.g., 3 variables: temperature, pressure, catalyst).
2. Click **Generate Initial Points**.
3. Choose **LHS** and set **Number of Points** to 10.
4. Click **Generate**.
5. Review the generated points in the experiment table and save to file if desired.

---

For more details on experiment management and data loading, see the next section of the workflow documentation.