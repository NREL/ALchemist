# BoTorch Acquisition Functions

The **BoTorch** backend in ALchemist provides a flexible and powerful interface for selecting the next experiment(s) using a variety of acquisition functions from the [BoTorch](https://botorch.org/) library. This guide explains the available options, how to use them, and what each setting means.

---

## Overview

The Acquisition panel in ALchemist allows you to:

- Choose between **Regular**, **Batch**, and **Exploratory** acquisition strategies.
- Select from several acquisition functions, each balancing exploration and exploitation in different ways.
- Customize parameters such as batch size and Monte Carlo integration points.
- Run the selected strategy to suggest the next experiment(s) based on your trained model.

---

## Important Note

You must first train your model using the BoTorch backend before running any BoTorch acquisition functions.  
See [BoTorch Backend](../modeling/botorch.md) for details on model training.

---

## Acquisition Types

### 1. Regular Acquisition

- **Expected Improvement (EI):**  
  Suggests points with the highest expected improvement over the current best observed value.
- **Log Expected Improvement (LogEI):**  
  Numerically stable version of EI.
- **Probability of Improvement (PI):**  
  Selects points with the highest probability of improving over the current best value.
- **Log Probability of Improvement (LogPI):**  
  Numerically stable version of PI.
- **Upper Confidence Bound (UCB):**  
  Balances exploration and exploitation by selecting points with the highest upper confidence bound.

**Customization:**  
- Choose to **maximize** or **minimize** your objective.

### 2. Batch Acquisition

- **q-Expected Improvement (qEI):**  
  Selects a batch of points that together maximize expected improvement.
- **q-Upper Confidence Bound (qUCB):**  
  Batch version of UCB.

**Customization:**  
- Set **batch size** (number of points to suggest at once, q).
- Monte Carlo samples (mc_samples) are used internally for batch methods.

### 3. Exploratory Acquisition

- **Integrated Posterior Variance (qNIPV):**  
  Selects points to reduce overall model uncertainty, focusing on exploration rather than optimization.

**Customization:**  
- Set the number of **Monte Carlo integration points** (higher values improve accuracy but increase computation time; 500–2000 is typical).

---

## How to Use

1. **Train Model:**  
   Train your model using the BoTorch backend. See [BoTorch Backend](../modeling/botorch.md) for instructions.

2. **Open Acquisition Panel:**  
   Go to the Acquisition panel. The BoTorch options will appear automatically.

3. **Choose Acquisition Type:**  
   Use the segmented button to select Regular, Batch, or Exploratory.

4. **Configure Options:**  
   - Select the acquisition function from the dropdown.
   - Adjust parameters (batch size, MC points) as needed.
   - Choose whether to maximize or minimize.

5. **Run Acquisition:**  
   Click **Run Acquisition Strategy** to suggest the next experiment(s). Results, including predicted value and uncertainty, will be shown in a notification window and highlighted in the data table and plots.

---

## Model Optimum Finder

In addition to acquisition functions, you can use the **Model Prediction Optimum** tool to find the point where the model predicts the best value (maximum or minimum).  
**Note:** This does not balance exploration and exploitation—it simply finds the model's optimum prediction.

---

## Tips & Notes

- **Batch Acquisition:** Use batch mode to suggest multiple experiments at once, useful for parallel experimentation.
- **Exploratory Mode:** Use qNIPV when you want to reduce model uncertainty rather than optimize the objective.
- **Parameter Tuning:** Increase MC points for more accurate but slower exploratory acquisition.
- **Publication Quality:** All results and suggested points are integrated with ALchemist's visualization tools for easy analysis and export.

---

For more details on the underlying algorithms, see the [BoTorch documentation](https://botorch.org/docs/acquisition.html)