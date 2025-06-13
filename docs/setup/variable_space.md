# Setting Up the Variable Space

The **Variable Space Setup** window allows you to define the variables that make up your search space for active learning experiments. This user guide explains each part of the dialog and how to use it.

---

## Opening the Variable Space Setup

- Click the **Define Search Space** button in the main application window to open the setup dialog.

---

## Window Overview

The window is divided into two main sections:

- **Left Panel:** Displays a list of variables you have defined.
- **Right Panel:** Contains control buttons for managing variables.

---

## Adding and Editing Variables

Each variable is represented as a row with the following fields:

- **Variable Name:** Enter a unique name for your variable.
- **Type:** Choose the variable type from the dropdown:
  - `Real`: For continuous variables (e.g., floating-point numbers).
  - `Integer`: For discrete variables (e.g., whole numbers).
  - `Categorical`: For variables with a set of named categories.

Depending on the type selected:
- **Real/Integer:** Enter minimum and maximum values in the `Min` and `Max` fields.
- **Categorical:** Click **Edit Values** to open a dialog where you can enter possible category values (one per row). Click **Save** in the dialog to confirm.

---

## Managing Variable Rows

Use the buttons on the right panel to manage your variable list:

- **Add Variable:** Add a new variable row.
- **Delete Row:** Remove the currently selected variable.
- **Clear Row:** Clear all fields in the selected row.
- **Move Up/Down:** Change the order of variables by moving the selected row up or down.

Click on a row to select it; the selected row is highlighted.

---

## Saving and Loading

- **Save to File:** Save your variable definitions to a `.json` or `.csv` file.
- **Load from File:** Load variable definitions from a `.json` or `.csv` file.
- **Save & Close:** Save your current variable space and close the window. This also updates the application's internal search space.

---

## Tips

- All fields must be filled out for a variable to be valid.
- For categorical variables, you must specify at least one value.
- You can reorder variables to control their order in the search space.

---

## Example

| Variable Name | Type        | Min | Max | Values           |
|---------------|-------------|-----|-----|------------------|
| temperature   | Real        | 20  | 100 |                  |
| batch_size    | Integer     | 1   | 10  |                  |
| catalyst      | Categorical |     |     | A, B, C, D       |

---

For more details on how the variable space is used, see the rest of the workflow documentation.