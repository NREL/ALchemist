import customtkinter as ctk
from customtkinter import filedialog
import json, csv, os
from skopt.space import Real, Integer, Categorical
from tkinter import StringVar
import tkinter.messagebox as messagebox
from tksheet import Sheet  # requires tksheet package

# -----------------------------------------------------------------------------
# A new toplevel for editing categorical values using tksheet.
class CategoricalEditorWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_values=None, callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.title("Edit Categorical Values")
        self.geometry("300x300")
        self.lift()
        self.focus_force()
        self.grab_set()
        self.callback = callback  # function to call with the new list
        
        # If there are initial values, use them; otherwise, create 10 blank rows.
        if initial_values and len(initial_values) > 0:
            data = [[v] for v in initial_values] + [[""] for _ in range(10 - len(initial_values))]
        else:
            data = [[""] for _ in range(10)]

        
        # Create a Sheet widget with one column.
        self.sheet = Sheet(self, data=data, headers=["Value"])
        self.sheet.enable_bindings()  # enable editing and navigation
        self.sheet.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # A save button to commit changes.
        btn_save = ctk.CTkButton(self, text="Save", command=self.on_save)
        btn_save.grid(row=1, column=0, pady=10)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
    
    def on_save(self):
        # Get all data from the sheet; expect a list of lists (one column each)
        raw_data = self.sheet.get_sheet_data()
        # Flatten and filter out empty strings.
        values = [row[0].strip() for row in raw_data if row and row[0].strip()]
        if self.callback:
            self.callback(values)
        self.destroy()

# -----------------------------------------------------------------------------
# Row widget representing a single variable.
class SpaceVariableRow(ctk.CTkFrame):
    def __init__(self, master, select_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.select_callback = select_callback
        self.configure(border_color="gray", border_width=1, corner_radius=5)
        
        # Initialize storage for categorical values.
        self.categorical_values = []
        
        # Entry for variable name.
        self.var_name_entry = ctk.CTkEntry(self, placeholder_text="Variable Name")
        self.var_name_entry.grid(row=0, column=0, padx=5, pady=5)
        self.var_name_entry.bind("<FocusIn>", lambda e: self.on_click(e))
        
        # Dropdown for selecting type.
        self.type_var = StringVar()
        self.type_dropdown = ctk.CTkOptionMenu(
            self,
            variable=self.type_var,
            values=["Real", "Integer", "Categorical"],
            command=self.on_type_change
        )
        self.type_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.type_var.set("Real")
        # (Do not bind the OptionMenu so its dropdown works correctly.)
        
        # For Real type: entries for min and max.
        self.min_entry = ctk.CTkEntry(self, placeholder_text="Min")
        self.max_entry = ctk.CTkEntry(self, placeholder_text="Max")
        self.min_entry.grid(row=0, column=2, padx=5, pady=5)
        self.max_entry.grid(row=0, column=3, padx=5, pady=5)
        self.min_entry.bind("<FocusIn>", lambda e: self.on_click(e))
        self.max_entry.bind("<FocusIn>", lambda e: self.on_click(e))
        
        # For Categorical type: a button to open the editor.
        self.edit_button = ctk.CTkButton(self, text="Edit Values", command=self.open_categorical_editor)
        self.edit_button.bind("<FocusIn>", lambda e: self.on_click(e))
        # Not gridded by default.
        
        # Bind the row itself (empty areas) so that clicking selects the row.
        self.bind("<Button-1>", self.on_click)
    
    def on_click(self, event):
        if self.select_callback:
            self.select_callback(self)
    
    def on_type_change(self, selection):
        if selection in ["Real", "Integer"]:
            self.min_entry.grid(row=0, column=2, padx=5, pady=5)
            self.max_entry.grid(row=0, column=3, padx=5, pady=5)
            self.edit_button.grid_forget()
        elif selection == "Categorical":
            self.min_entry.grid_forget()
            self.max_entry.grid_forget()
            self.edit_button.grid(row=0, column=2, padx=5, pady=5, columnspan=2)
    
    def open_categorical_editor(self):
        # Open the CategoricalEditorWindow, passing current values.
        CategoricalEditorWindow(self, initial_values=self.categorical_values, callback=self.set_categorical_values)
    
    def set_categorical_values(self, values):
        self.categorical_values = values
        # Update the button text to show how many values are set.
        self.edit_button.configure(text=f"Edit Values ({len(values)})")
    
    def get_data(self):
        name = self.var_name_entry.get().strip()
        if not name:
            return None
        typ = self.type_var.get()
        if typ in ["Real", "Integer"]:
            try:
                min_val = float(self.min_entry.get())
                max_val = float(self.max_entry.get())
            except ValueError:
                return None
            return {"name": name, "type": typ, "min": min_val, "max": max_val}
        elif typ == "Categorical":
            if not self.categorical_values:
                return None
            return {"name": name, "type": "Categorical", "values": self.categorical_values}
    
    def populate(self, data):
        self.var_name_entry.delete(0, ctk.END)
        self.var_name_entry.insert(0, data.get("name", ""))
        self.type_var.set(data.get("type", "Real"))
        self.on_type_change(data.get("type", "Real"))
        if data.get("type") == "Real":
            self.min_entry.delete(0, ctk.END)
            self.min_entry.insert(0, str(data.get("min", "")))
            self.max_entry.delete(0, ctk.END)
            self.max_entry.insert(0, str(data.get("max", "")))
        elif data.get("type") == "Categorical":
            self.categorical_values = data.get("values", [])
            self.edit_button.configure(text=f"Edit Values ({len(self.categorical_values)})")
    
    def clear(self):
        self.var_name_entry.delete(0, ctk.END)
        self.min_entry.delete(0, ctk.END)
        self.max_entry.delete(0, ctk.END)
        self.categorical_values = []
        self.edit_button.configure(text="Edit Values")
    
    def set_selected(self, selected: bool):
        if selected:
            self.configure(border_color="blue", border_width=2)
        else:
            self.configure(border_color="gray", border_width=1)

# -----------------------------------------------------------------------------
# Top-level window for setting up the variable space.
class SpaceSetupWindow(ctk.CTkToplevel):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Search Space Setup")
        self.geometry("800x500")
        self.selected_row = None  # currently selected row
        self.categorical_variables = []  # Store categorical variable names

        # Main container with left (rows) and right (controls) panels.
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel: header and scrollable area.
        left_frame = ctk.CTkFrame(main_container)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Header using pack.
        header = ctk.CTkFrame(left_frame)
        header.pack(fill="x", padx=5, pady=(0, 5))
        lbl_name = ctk.CTkLabel(header, text="Variable Name", width=120)
        lbl_name.pack(side="left", padx=5)
        lbl_type = ctk.CTkLabel(header, text="Type", width=120)
        lbl_type.pack(side="left", padx=5)
        lbl_params = ctk.CTkLabel(header, text="Parameters", width=240)
        lbl_params.pack(side="left", padx=5)
        
        # Scrollable frame for variable rows.
        self.scrollable_frame = ctk.CTkScrollableFrame(left_frame, height=350)
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.rows_frame = ctk.CTkFrame(self.scrollable_frame)
        self.rows_frame.pack(fill="x", expand=True)
        self.rows = []
        
        # Right panel: control buttons.
        control_panel = ctk.CTkFrame(main_container, width=150)
        control_panel.pack(side="right", fill="y")
        btn_add = ctk.CTkButton(control_panel, text="Add Variable", command=self.add_row)
        btn_add.pack(pady=5, fill="x")
        btn_delete = ctk.CTkButton(control_panel, text="Delete Row", command=self.delete_selected_row)
        btn_delete.pack(pady=5, fill="x")
        btn_clear = ctk.CTkButton(control_panel, text="Clear Row", command=self.clear_selected_row)
        btn_clear.pack(pady=5, fill="x")
        btn_move_up = ctk.CTkButton(control_panel, text="Move Up", command=self.move_selected_up)
        btn_move_up.pack(pady=5, fill="x")
        btn_move_down = ctk.CTkButton(control_panel, text="Move Down", command=self.move_selected_down)
        btn_move_down.pack(pady=5, fill="x")
        btn_load = ctk.CTkButton(control_panel, text="Load from File", command=self.load_from_file)
        btn_load.pack(pady=5, fill="x")
        btn_save_file = ctk.CTkButton(control_panel, text="Save to File", command=self.save_to_file)
        btn_save_file.pack(pady=5, fill="x")
        btn_save_close = ctk.CTkButton(control_panel, text="Save & Close", command=self.save_and_close)
        btn_save_close.pack(pady=5, fill="x")
        
        # Load existing data if available; otherwise, add one row.
        if hasattr(self.master, "variable_space_data") and self.master.variable_space_data:
            self.load_data(self.master.variable_space_data)
        else:
            self.add_row()
        
        # Bring this window to the front.
        self.lift()
        self.focus_force()
        self.grab_set()
    
    def add_row(self):
        row = SpaceVariableRow(self.rows_frame, select_callback=self.select_row)
        row.pack(fill="x", pady=2)
        self.rows.append(row)
    
    def select_row(self, row_widget):
        if self.selected_row:
            self.selected_row.set_selected(False)
        self.selected_row = row_widget
        self.selected_row.set_selected(True)
    
    def refresh_rows(self):
        for row in self.rows:
            row.pack_forget()
        for row in self.rows:
            row.pack(fill="x", pady=2)
    
    def move_selected_up(self):
        if self.selected_row is None:
            return
        idx = self.rows.index(self.selected_row)
        if idx > 0:
            self.rows[idx], self.rows[idx-1] = self.rows[idx-1], self.rows[idx]
            self.refresh_rows()
    
    def move_selected_down(self):
        if self.selected_row is None:
            return
        idx = self.rows.index(self.selected_row)
        if idx < len(self.rows) - 1:
            self.rows[idx], self.rows[idx+1] = self.rows[idx+1], self.rows[idx]
            self.refresh_rows()
    
    def delete_selected_row(self):
        if self.selected_row is None:
            return
        self.rows.remove(self.selected_row)
        self.selected_row.destroy()
        self.selected_row = None
    
    def clear_selected_row(self):
        if self.selected_row is None:
            return
        self.selected_row.clear()
    
    def get_data(self):
        data_list = []
        for row in self.rows:
            d = row.get_data()
            if d is not None:
                data_list.append(d)
        return data_list
    
    def load_data(self, data_list):
        for row in self.rows:
            row.destroy()
        self.rows = []
        for d in data_list:
            row = SpaceVariableRow(self.rows_frame, select_callback=self.select_row)
            row.populate(d)
            row.pack(fill="x", pady=2)
            self.rows.append(row)
    
    def save_to_file(self):
        data = self.get_data()
        if not data:
            messagebox.showerror("Error", "No valid variable data to save.")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save Variable Space",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("CSV files", "*.csv"))
        )
        if not file_path:
            return
        try:
            if file_path.lower().endswith(".json"):
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)
            elif file_path.lower().endswith(".csv"):
                with open(file_path, "w", newline="") as f:
                    fieldnames = ["Variable", "Type", "Min", "Max", "Values"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for d in data:
                        row_data = {
                            "Variable": d.get("name", ""),
                            "Type": d.get("type", ""),
                            "Min": d.get("min", ""),
                            "Max": d.get("max", ""),
                            "Values": ", ".join(d.get("values", [])) if d.get("type") == "Categorical" else ""
                        }
                        writer.writerow(row_data)
            messagebox.showinfo("Success", f"Saved variable space to:\n{os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
    
    def load_from_file(self):
        file_path = filedialog.askopenfilename(
            title="Load Variable Space",
            filetypes=(("JSON files", "*.json"), ("CSV files", "*.csv"))
        )
        if not file_path:
            return
        try:
            if file_path.lower().endswith(".json"):
                with open(file_path, "r") as f:
                    data = json.load(f)
            elif file_path.lower().endswith(".csv"):
                data = []
                with open(file_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        typ = row.get("Type", "").strip()
                        if typ == "Real":
                            d = {
                                "name": row.get("Variable", "").strip(),
                                "type": "Real",
                                "min": float(row.get("Min", 0)),
                                "max": float(row.get("Max", 0))
                            }
                        elif typ == "Integer":
                            d = {
                                "name": row.get("Variable", "").strip(),
                                "type": "Integer",
                                "min": 0,
                                "max": 1
                            }
                        elif typ == "Categorical":
                            values = [v.strip() for v in row.get("Values", "").split(",") if v.strip()]
                            d = {
                                "name": row.get("Variable", "").strip(),
                                "type": "Categorical",
                                "values": values
                            }
                        else:
                            continue
                        data.append(d)
            self.load_data(data)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
    
    def save_and_close(self):
        """Save the search space and update the categorical variables list."""
        data = self.get_data()
        if not data:
            messagebox.showerror("Error", "No valid variable data to save.")
            return
        self.master.variable_space_data = data
        self.search_space = []
        self.categorical_variables = []  # Reset the list

        for d in data:
            if d["type"] == "Real":
                self.search_space.append(Real(d["min"], d["max"], name=d["name"]))
            elif d["type"] == "Integer":
                self.search_space.append(Integer(d["min"], d["max"], name=d["name"]))
            elif d["type"] == "Categorical":
                self.search_space.append(Categorical(d["values"], name=d["name"]))
                self.categorical_variables.append(d["name"])  # Add to categorical list

        print("Saved search space:")
        for obj in self.search_space:
            print(obj)
        print("Categorical variables:", self.categorical_variables)
        self.destroy()

# -----------------------------------------------------------------------------
# Main application window.
if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = ctk.CTk()
    app.geometry("600x400")
    app.title("Main Application")
    
    # The main app holds the variable space data.
    app.variable_space_data = None
    
    def open_space_setup():
        SpaceSetupWindow(app)
    
    open_btn = ctk.CTkButton(app, text="Define Search Space", command=open_space_setup)
    open_btn.pack(pady=20)
    
    app.mainloop()
