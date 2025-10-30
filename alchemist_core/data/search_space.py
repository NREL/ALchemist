from typing import List, Dict, Any, Union, Optional
from skopt.space import Real, Integer, Categorical
import numpy as np
import pandas as pd
import json

class SearchSpace:
    """
    Class for storing and managing the search space in a consistent way across backends.
    Provides methods for conversions to different formats required by different backends.
    """
    def __init__(self):
        self.variables = []  # List of variable dictionaries with metadata
        self.skopt_dimensions = []  # skopt dimensions (used by scikit-learn)
        self.categorical_variables = []  # List of categorical variable names

    def add_variable(self, name: str, var_type: str, **kwargs):
        """
        Add a variable to the search space.
        
        Args:
            name: Variable name
            var_type: "real", "integer", or "categorical"
            **kwargs: Additional parameters (min, max, values)
        """
        var_dict = {"name": name, "type": var_type.lower()}
        var_dict.update(kwargs)
        self.variables.append(var_dict)
        
        # Create the corresponding skopt dimension
        if var_type.lower() == "real":
            self.skopt_dimensions.append(Real(kwargs["min"], kwargs["max"], name=name))
        elif var_type.lower() == "integer":
            self.skopt_dimensions.append(Integer(kwargs["min"], kwargs["max"], name=name))
        elif var_type.lower() == "categorical":
            self.skopt_dimensions.append(Categorical(kwargs["values"], name=name))
            self.categorical_variables.append(name)
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    def from_dict(self, data: List[Dict[str, Any]]):
        """Load search space from a list of dictionaries (used with JSON/CSV loading)."""
        self.variables = []
        self.skopt_dimensions = []
        self.categorical_variables = []
        
        for var in data:
            var_type = var["type"].lower()
            if var_type in ["real", "integer"]:
                self.add_variable(
                    name=var["name"],
                    var_type=var_type,
                    min=var["min"],
                    max=var["max"]
                )
            elif var_type == "categorical":
                self.add_variable(
                    name=var["name"],
                    var_type=var_type,
                    values=var["values"]
                )
        
        return self

    def from_skopt(self, dimensions):
        """Load search space from skopt dimensions."""
        self.variables = []
        self.skopt_dimensions = dimensions.copy()
        self.categorical_variables = []
        
        for dim in dimensions:
            name = dim.name
            if isinstance(dim, Real):
                self.variables.append({
                    "name": name,
                    "type": "real",
                    "min": dim.low,
                    "max": dim.high
                })
            elif isinstance(dim, Integer):
                self.variables.append({
                    "name": name,
                    "type": "integer",
                    "min": dim.low,
                    "max": dim.high
                })
            elif isinstance(dim, Categorical):
                self.variables.append({
                    "name": name,
                    "type": "categorical",
                    "values": list(dim.categories)
                })
                self.categorical_variables.append(name)
        
        return self

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert search space to a list of dictionaries."""
        return self.variables.copy()

    def to_skopt(self) -> List[Union[Real, Integer, Categorical]]:
        """Get skopt dimensions for scikit-learn."""
        return self.skopt_dimensions.copy()

    def to_ax_space(self) -> Dict[str, Dict[str, Any]]:
        """Convert to Ax parameter format."""
        ax_params = {}
        for var in self.variables:
            name = var["name"]
            if var["type"] == "real":
                ax_params[name] = {
                    "name": name,
                    "type": "range",
                    "bounds": [var["min"], var["max"]],
                }
            elif var["type"] == "integer":
                ax_params[name] = {
                    "name": name,
                    "type": "range",
                    "bounds": [var["min"], var["max"]],
                    "value_type": "int",
                }
            elif var["type"] == "categorical":
                ax_params[name] = {
                    "name": name,
                    "type": "choice",
                    "values": var["values"],
                }
        return ax_params

    def to_botorch_bounds(self) -> Dict[str, np.ndarray]:
        """Create bounds in BoTorch format."""
        bounds = {}
        for var in self.variables:
            if var["type"] in ["real", "integer"]:
                bounds[var["name"]] = np.array([var["min"], var["max"]])
        return bounds

    def get_variable_names(self) -> List[str]:
        """Get list of all variable names."""
        return [var["name"] for var in self.variables]

    def get_categorical_variables(self) -> List[str]:
        """Get list of categorical variable names."""
        return self.categorical_variables.copy()

    def get_integer_variables(self) -> List[str]:
        """Get list of integer variable names."""
        return [var["name"] for var in self.variables if var["type"] == "integer"]

    def save_to_json(self, filepath: str):
        """Save search space to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_json(self, filepath: str):
        """Load search space from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return self.from_dict(data)
    
    @classmethod
    def from_json(cls, filepath: str):
        """Class method to create a SearchSpace from a JSON file."""
        instance = cls()
        return instance.load_from_json(filepath)

    def __len__(self):
        return len(self.variables)
