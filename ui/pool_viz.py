"""
DEPRECATED: Pool Visualization Utilities

This module contains legacy pool generation and visualization functions.

WARNING: These functions are deprecated and will be removed in a future version.
The pool-based visualization approach is no longer used in modern acquisition
strategies. The acquisition functions now work directly with the search space
without requiring pre-generated candidate pools.

Legacy behavior:
- generate_pool(): Creates a large LHS sample of candidate points
- plot_pool(): Visualizes the candidate pool as a scatter plot

Modern approach (alchemist_core):
- Acquisition functions optimize directly over the search space
- No pre-generated pools needed
- More efficient and flexible

For new code, use:
- OptimizationSession.suggest_next() from alchemist_core
- Visualization through ui/visualizations.py

This module will be removed in v0.3.0
"""

import csv
import pandas as pd
import numpy as np
import warnings
import hashlib
import os
import joblib
import json
from skopt.space import Real, Integer, Categorical
from skopt.sampler import Lhs


def search_space_to_dict_list(search_space):
    """
    DEPRECATED: Convert a list of skopt.space objects into a list of dictionaries.
    
    This is only used for pool caching, which is deprecated.
    """
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
    DEPRECATED: Generate a pool of experimental candidate points using LHS sampling.
    
    WARNING: This function is deprecated. Modern acquisition strategies in 
    alchemist_core do not require pre-generated pools.
    
    Args:
        search_space: List of skopt.space objects defining the search space
        experiments_df: Optional DataFrame of existing experiments to append
        pool_size: Number of candidate points to generate (max 10000)
        lhs_iterations: Number of LHS optimization iterations (max 20)
        cache_dir: Directory to cache generated pools
        debug: Enable debug logging
    
    Returns:
        pd.DataFrame: Pool of candidate experimental points
    
    Note:
        This function will be removed in v0.3.0. Use OptimizationSession.suggest_next()
        instead, which optimizes directly over the search space.
    """
    warnings.warn(
        "generate_pool() is deprecated and will be removed in v0.3.0. "
        "Modern acquisition functions in alchemist_core do not require pools. "
        "Use OptimizationSession.suggest_next() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if debug:
        print("Generating pool (DEPRECATED)...")

    # Cap pool_size and lhs_iterations to recommended limits
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

    if debug:
        print("Search space definition:")
        for dim in search_space:
            print(f"  - {dim.name}: {dim}")

    # Generate a cache key based on the search space and sampling parameters
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
        
        if debug and "Catalyst" in pool.columns:
            print(f"Unique Catalyst values in cached pool: {pool['Catalyst'].unique()}")
        elif debug:
            print("Catalyst column not found in cached pool!")

        return pool

    # Extract variable names from the search space
    var_names = [dim.name for dim in search_space]

    warnings.warn(
        "A new pool of experimental points is being generated. Repeating this process may lead to inconsistent sampling.\n"
        "To ensure consistent optimization, make sure the cache is saved in the correct cache directory for reuse.",
        UserWarning
    )

    sampler = Lhs(lhs_type="classic", criterion="maximin", iterations=lhs_iterations)
    sampled_points = sampler.generate(search_space, pool_size)

    # Convert the list of sampled dictionaries into a DataFrame
    sampled_df = pd.DataFrame(sampled_points, columns=var_names)

    if debug and "Catalyst" in sampled_df.columns:
        print(f"Unique Catalyst values in sampled_df: {sampled_df['Catalyst'].unique()}")
    elif debug:
        print("Catalyst column missing from sampled_df!")

    # If there are existing experiments, append them
    if experiments_df is not None and not experiments_df.empty:
        if debug:
            print("Appending existing experiments to the pool.")
        existing_points = experiments_df.drop(columns='Output').values.astype(float)
        existing_df = pd.DataFrame(existing_points, columns=var_names)
        pool = pd.concat([sampled_df, existing_df], ignore_index=True)
    else:
        pool = sampled_df

    if debug and "Catalyst" in pool.columns:
        print(f"Final unique Catalyst values in pool: {pool['Catalyst'].unique()}")
    elif debug:
        print("Catalyst column missing from final pool!")

    # Cache the pool for future reuse
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    joblib.dump(pool, cache_file)
    if debug:
        print("Saving new pool to cache.")

    return pool


def plot_pool(pool, var1, var2, ax, kmeans=None, add_cluster=False, experiments=None):
    """
    DEPRECATED: Plot a scatter plot of candidate pool points.
    
    WARNING: This visualization is deprecated. Modern acquisition visualizations
    should use ui/visualizations.py functions instead.
    
    Args:
        pool: DataFrame containing the candidate points
        var1: Name of the variable for the x-axis
        var2: Name of the variable for the y-axis
        ax: Matplotlib axis object to plot on
        kmeans: Optional clustering object (deprecated)
        add_cluster: Whether to highlight largest empty cluster (deprecated)
        experiments: Optional experiment data (unused)
    
    Note:
        This function will be removed in v0.3.0.
    """
    warnings.warn(
        "plot_pool() is deprecated and will be removed in v0.3.0. "
        "Use visualization functions from ui/visualizations.py instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Extract the data for the selected variables
    x_data = pool[var1]
    y_data = pool[var2]

    if kmeans is not None:
        labels = kmeans.labels_
        for i in range(kmeans.n_clusters):
            cluster_points = pool[labels == i]
            ax.scatter(cluster_points[var1], cluster_points[var2],
                       label=f'Cluster {i}', alpha=0.1)
        
        # If requested and available, highlight the largest empty cluster
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
    ax.set_title("Experimental Pool (DEPRECATED)")
