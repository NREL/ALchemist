# Loading Experimental Data

The **Experiment Data** panel in ALchemist lets you load, view, and manage your experimental results. This is a key step before training surrogate models or running active learning.

---

## Loading Data from File

1. **Click "Load Experiments":**  
   In the Experiment Data panel, click the **Load Experiments** button.
2. **Select Your File:**  
   Choose a `.csv` file containing your experimental data. The file should have columns for each variable (matching your variable space) and an `Output` column for the measured result. Optionally, you can include a `Noise` column to specify measurement uncertainty for each point.
3. **Data Appears in the Table:**  
   The loaded data will be displayed in the table. If a `Noise` column is present, it will be used for model regularization.

**Tip:**  
If your data columns do not match the variable names or required format, you may see an error. Make sure your CSV headers match your variable space exactly.

---

## Adding a New Experiment Point

You can add a new experiment directly from the UI:

1. **Click "Add Point":**  
   Opens a dialog where you can enter values for each variable, the output, and (optionally) the noise.
2. **Fill in the Fields:**  
   Enter values for all variables and the output. If you know the measurement uncertainty, enter it in the Noise field.
3. **Save & Close:**  
   Click **Save & Close** to add the point to your experiment table. You can also choose to save the updated data to file and retrain the model immediately by checking the corresponding boxes.

**Note:**  
- There may be issues with type compatibility (e.g., numbers being saved as strings). If you encounter problems, check your CSV file and ensure numeric columns are formatted correctly.
- Sometimes, changes made directly in the table (tksheet widget) may not update the internal experiment data until you save or reload. Use the provided dialogs for best results.

---

## Saving Your Data

- **Click "Save Experiments":**  
  Saves the current experiment table to a `.csv` file.  
- **Tip:**  
  Always save your data before closing the application to avoid losing changes.

---

## Retraining the Model

- When adding a new point, you can check **Retrain model** to automatically update the surrogate model with the new data.
- If retraining does not seem to trigger, you may need to retrain manually from the model panel.

---

## Known Issues & Tips

- **Type Compatibility:**  
  Data entered via the table or add-point dialog may sometimes be interpreted as strings. If you see errors or unexpected behavior, check your data types in the CSV file.
- **Table Edits:**  
  Editing data directly in the table does not always update the internal experiment manager. For reliable results, use the add-point dialog or reload your data after editing.
- **Noise Column:**  
  The noise column is optional. If present, it should be numeric. You can toggle its visibility in the Preferences menu.

---

For more details on managing experiments and troubleshooting, see the rest of the workflow documentation.