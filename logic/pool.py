import csv
import pandas as pd
import numpy as np
import warnings
import hashlib
import os
import joblib
import json
from CTkMessagebox import CTkMessagebox
from skopt.space import Real, Integer, Categorical
from skopt.sampler import Lhs
from logic.clustering import cluster_pool

def search_space_to_dict_list(search_space):
    """Convert a list of skopt.space objects into a list of dictionaries for hashing/caching."""
    dict_list = []
    for dim in search_space:
        if isinstance(dim, Categorical):
            dict_list.append({
                "name": dim.name,
                "type": "Categorical",
                "values": dim.categories
            })
        elif isinstance(dim, Real):
            dict_list.append({
                "name": dim.name,
                "type": "Real",
                "low": dim.low,
                "high": dim.high
            })
        elif isinstance(dim, Integer):
            dict_list.append({
                "name": dim.name,
                "type": "Integer",
                "low": dim.low,
                "high": dim.high
            })
    return dict_list



def generate_pool(search_space, experiments_df=None, pool_size=10000, lhs_iterations=5, cache_dir="cache", debug=False):
    """
    Generates a pool of experimental points using the given search_space (a list of skopt.space objects)
    with caching support.
    """
    if debug:
        print("Generating pool...")

    # Cap pool_size and lhs_iterations to recommended limits.
    if pool_size > 10000:
        warnings.warn(
            f"Pool size of {pool_size} exceeds the recommended maximum of 10,000. "
            "Automatically reducing to 10,000 for better performance.",
            RuntimeWarning
        )
        pool_size = 10000

    if lhs_iterations > 20:
        warnings.warn(
            f"{lhs_iterations} iterations for LHS exceeds the recommended maximum of 20. "
            "Automatically reducing to 20 for better performance.",
            RuntimeWarning
        )
        lhs_iterations = 20

    # Debug print: Check search space definition
    if debug:
        print("Search space definition:")
        for dim in search_space:
            print(f"  - {dim.name}: {dim}")

    # Generate a cache key based on the search space and sampling parameters.
    space_dict = search_space_to_dict_list(search_space)
    hash_input = json.dumps(space_dict) + f"_{pool_size}_{lhs_iterations}"
    if experiments_df is not None and not experiments_df.empty:
        hash_input += f"_{experiments_df.to_json()}"
    cache_key = hashlib.md5(hash_input.encode()).hexdigest()

    cache_file = os.path.join(cache_dir, f"pool_{cache_key}.pkl")
    if os.path.exists(cache_file):
        if debug:
            print("Loading cached pool.")
        pool = joblib.load(cache_file)
        
        # Debug print: Check if SAPO34 is in the loaded pool
        if debug and "Catalyst" in pool.columns:
            print("Unique Catalyst values in cached pool:", pool["Catalyst"].unique())
        elif debug:
            print("Warning: Catalyst column not found in cached pool!")

        return pool

    # Extract variable names from the search space.
    var_names = [dim.name for dim in search_space]

    warnings.warn(
        "A new pool of experimental points is being generated. Repeating this process may lead to inconsistent sampling.\n"
        "To ensure consistent optimization, make sure the cache is saved in the correct cache directory for reuse.",
        UserWarning
    )

    sampler = Lhs(lhs_type="classic", criterion="maximin", iterations=lhs_iterations)
    sampled_points = sampler.generate(search_space, pool_size)

    # Convert the list of sampled dictionaries into a DataFrame.
    sampled_df = pd.DataFrame(sampled_points, columns=var_names)

    # Debug print: Check the unique values sampled for categorical variables
    if debug and "Catalyst" in sampled_df.columns:
        print("Unique Catalyst values in sampled_df:", sampled_df["Catalyst"].unique())
    elif debug:
        print("Warning: Catalyst column missing from sampled_df!")

    # If there are existing experiments, append them.
    if experiments_df is not None and not experiments_df.empty:
        if debug:
            print("Appending existing experiments to the pool.")
        existing_points = experiments_df.drop(columns='Output').values.astype(float)
        existing_df = pd.DataFrame(existing_points, columns=var_names)
        pool = pd.concat([sampled_df, existing_df], ignore_index=True)
    else:
        pool = sampled_df

    # Debug print: Final unique values after merging with experiments
    if debug and "Catalyst" in pool.columns:
        print("Final unique Catalyst values in pool:", pool["Catalyst"].unique())
    elif debug:
        print("Warning: Catalyst column missing from final pool!")

    # Cache the pool for future reuse.
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    joblib.dump(pool, cache_file)
    if debug:
        print("Saving new pool to cache.")

    return pool


def load_search_space_from_file(file_path):
    """
    Load a search space from a JSON or CSV file.
    Returns a tuple: (search_space, categorical_variables).
    """
    try:
        # Load data from the file
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
                            "min": int(row.get("Min", 0)),
                            "max": int(row.get("Max", 0))
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

        # Convert the dictionary representation into skopt.space objects
        search_space = []
        categorical_variables = []  # Track categorical variable names
        for d in data:
            if d["type"] == "Real":
                search_space.append(Real(d["min"], d["max"], name=d["name"]))
            elif d["type"] == "Integer":
                search_space.append(Integer(d["min"], d["max"], name=d["name"]))
            elif d["type"] == "Categorical":
                search_space.append(Categorical(d["values"], name=d["name"]))
                categorical_variables.append(d["name"])  # Add to categorical list

        return search_space, categorical_variables

    except Exception as e:
        CTkMessagebox.showerror("Load Error", str(e))
        return None, None

def plot_pool(pool, var1, var2, ax, kmeans=None, add_cluster=False, experiments=None):
    """
    Plots a scatter plot of two variables from the experimental pool.

    Args:
        pool (pd.DataFrame): DataFrame containing the experimental points.
        var1 (str): Name of the variable for the x-axis.
        var2 (str): Name of the variable for the y-axis.
        ax (matplotlib.axes.Axes): The axis object to plot the data on.
        kmeans: A precomputed clustering object. If None, clustering is not performed.
        add_cluster (bool): If True and kmeans has 'largest_empty_cluster', highlights that cluster.
        experiments: (Unused in this version; previously used for clustering.)
    """
    # Extract the data for the selected variables
    x_data = pool[var1]
    y_data = pool[var2]

    if kmeans is not None:
        labels = kmeans.labels_
        for i in range(kmeans.n_clusters):
            cluster_points = pool[labels == i]
            ax.scatter(cluster_points[var1], cluster_points[var2],
                       label=f'Cluster {i}', alpha=0.1)
        
        # If requested and available, highlight the largest empty cluster.
        if add_cluster and hasattr(kmeans, 'largest_empty_cluster'):
            largest_empty_cluster = kmeans.largest_empty_cluster
            largest_empty_cluster_points = pool[labels == largest_empty_cluster]
            ax.scatter(largest_empty_cluster_points[var1],
                       largest_empty_cluster_points[var2],
                       marker='o', alpha=0.9, label='Largest Empty Cluster')
    else:
        ax.scatter(x_data, y_data, alpha=0.1)
    
    # Set the labels and title of the plot
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title("Experimental Pool")
