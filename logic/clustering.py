from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from skopt.space import Categorical

'''
def cluster_pool(pool, experiments, add_cluster=True):
    """
    Clusters the pool and finds the largest empty cluster.

    Args:
        pool (pd.DataFrame or np.ndarray): The pool of candidate points.
        experiments (pd.DataFrame): Previously completed experiments (must contain an 'output' column).

    Returns:
        Tuple (np.ndarray, int, KMeans): Cluster labels for the pool, index of the empty cluster, and the KMeans object.
    """
    # Determine the number of completed experiments
    n_experiments = len(experiments)

    # Drop the 'output' column from experiments before clustering
    experiments_inputs = experiments.drop(columns=['Output'])

    # Set number of clusters
    n_clusters = n_experiments + 1 if add_cluster else n_experiments

    # Standardize both the pool and the experiments inputs
    scaler = StandardScaler()
    
    # Fit and transform the pool, and transform the experiments inputs
    pool_scaled = scaler.fit_transform(pool)
    experiments_scaled = scaler.transform(experiments_inputs)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pool_scaled)

    # Identify clusters that contain experimental points
    exp_labels = kmeans.predict(experiments_scaled) if not experiments.empty else []

    if add_cluster:
        # Find the largest "empty" cluster (i.e., a cluster with no previous experiments)
        empty_clusters = set(range(n_clusters)) - set(exp_labels)
        if not empty_clusters:
            raise ValueError("No empty cluster found!")  # Should never happen unless data is weird

        # Select the largest empty cluster
        largest_empty_cluster = max(empty_clusters, key=lambda c: (labels == c).sum())
        kmeans.largest_empty_cluster = largest_empty_cluster

        return labels, largest_empty_cluster, kmeans

    return labels, None, kmeans
'''
def cluster_pool(pool, experiments, search_space, add_cluster=True):
    """
    Clusters the pool and finds the largest empty cluster.

    Args:
        pool (pd.DataFrame): The pool of candidate points.
        experiments (pd.DataFrame): Previously completed experiments (must contain an 'Output' column).
        search_space (list): List of skopt.space objects defining the variable space.
        add_cluster (bool): If True, finds and returns the largest empty cluster.

    Returns:
        Tuple (np.ndarray, int, KMeans): Cluster labels for the pool, index of the empty cluster (if any), and the KMeans object.
    """
    # Determine the number of completed experiments
    n_experiments = len(experiments)

    # Identify categorical columns based on search_space
    cat_cols = [dim.name for dim in search_space if isinstance(dim, Categorical)]
    
    # One-hot encode categorical columns for the pool if needed.
    if cat_cols:
        encoded_pool = pd.get_dummies(pool, columns=cat_cols)
    else:
        encoded_pool = pool.copy()

    # Drop the 'Output' column from experiments before clustering.
    experiments_inputs = experiments.drop(columns=['Output'])
    
    # One-hot encode the experiments inputs, reindexing to match the pool's encoded columns.
    if cat_cols:
        experiments_encoded = pd.get_dummies(experiments_inputs, columns=cat_cols)
        experiments_encoded = experiments_encoded.reindex(columns=encoded_pool.columns, fill_value=0)
    else:
        experiments_encoded = experiments_inputs.copy()

    # Set the number of clusters.
    n_clusters = n_experiments + 1 if add_cluster else n_experiments

    # Standardize the encoded pool and experiments data.
    scaler = StandardScaler()
    pool_scaled = scaler.fit_transform(encoded_pool)
    experiments_scaled = scaler.transform(experiments_encoded)

    # Perform K-Means clustering on the scaled pool.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pool_scaled)

    # Identify clusters that contain experimental points.
    exp_labels = kmeans.predict(experiments_scaled) if not experiments.empty else []

    if add_cluster:
        # Find the largest "empty" cluster (i.e., no experimental point belongs to it).
        empty_clusters = set(range(n_clusters)) - set(exp_labels)
        if not empty_clusters:
            raise ValueError("No empty cluster found!")  # This should not happen unless data is unusual.

        # Select the empty cluster with the most pool points.
        largest_empty_cluster = max(empty_clusters, key=lambda c: (labels == c).sum())
        kmeans.largest_empty_cluster = largest_empty_cluster

        return labels, largest_empty_cluster, kmeans

    return labels, None, kmeans
