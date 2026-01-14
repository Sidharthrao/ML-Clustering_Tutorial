"""
Configuration module for E-commerce Customer Segmentation ML Pipeline.
Contains all configuration parameters, paths, and hyperparameters for clustering.
"""
import os
from pathlib import Path
from typing import Dict, Any, List

# Project root directory - absolute path to ECommerce behavior data
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Database path - relative to project root (goes up to Capstone_Projects level)
DATABASE_PATH = PROJECT_ROOT.parent.parent / "Database.db"

# Virtual environment path at repository root
REPO_ROOT = PROJECT_ROOT.parent.parent.parent.parent.parent
VENV_PATH = REPO_ROOT / "venv"

# Directory paths (relative to PROJECT_ROOT)
DIRECTORIES = {
    "models": PROJECT_ROOT / "models",
    "logs": PROJECT_ROOT / "logs",
    "reports": PROJECT_ROOT / "reports",
    "notebooks": PROJECT_ROOT / "notebooks",
    "data": PROJECT_ROOT / "data",
}

# Model file paths
MODEL_PATHS = {
    "preprocessor": DIRECTORIES["models"] / "preprocessor.pkl",
    "clusterer": DIRECTORIES["models"] / "clusterer.pkl",
    "feature_names": DIRECTORIES["models"] / "feature_names.pkl",
}

# Logging configuration
LOG_CONFIG = {
    "log_file": DIRECTORIES["logs"] / "ecommerce_clustering.log",
    "log_level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# Database configuration
DB_TABLE_NAME = "Ecommerce_data"

# Clustering algorithm configurations
CLUSTERING_CONFIG = {
    "random_state": 42,
    "n_jobs": -1,  # Use all available cores
}

# K-means configuration
KMEANS_CONFIG = {
    "n_clusters_range": list(range(2, 11)),  # Test 2 to 10 clusters
    "init": "k-means++",
    "n_init": 10,  # Number of initializations
    "max_iter": 300,
    "tol": 1e-4,
    "random_state": 42,
}

# DBSCAN configuration
DBSCAN_CONFIG = {
    "eps_range": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # Range of eps values to test
    "min_samples_range": [3, 5, 10, 15, 20],  # Range of min_samples to test
    "metric": "euclidean",
}

# Hierarchical clustering configuration
HIERARCHICAL_CONFIG = {
    "n_clusters_range": list(range(2, 11)),
    "linkage_options": ["ward", "complete", "average"],  # Different linkage methods
    "affinity": "euclidean",  # Distance metric
}

# GMM configuration
GMM_CONFIG = {
    "n_components_range": list(range(2, 11)),
    "covariance_type": "full",  # 'full', 'tied', 'diag', 'spherical'
    "n_init": 10,
}

# Spectral Clustering configuration
SPECTRAL_CONFIG = {
    "n_clusters_range": list(range(2, 11)),
    "affinity": "rbf",  # 'nearest_neighbors', 'rbf'
    "n_neighbors": 10,
    "assign_labels": "kmeans",
}

# HDBSCAN configuration
HDBSCAN_CONFIG = {
    "min_cluster_size_range": [5, 10, 15, 20],
    "min_samples_range": [None, 5, 10],  # None means same as min_cluster_size
    "cluster_selection_epsilon": 0.0,
    "metric": "euclidean",
}

# Stability Analysis configuration
STABILITY_CONFIG = {
    "n_bootstrap": 20, # Number of bootstrap samples
    "sample_fraction": 0.8, # Fraction of data to sample
    "threshold": 0.7, # Threshold for stable clusters
}

# Feature engineering configuration
FEATURE_CONFIG = {
    # RFM Analysis thresholds (percentile-based)
    "rfm_percentiles": {
        "recency": [25, 50, 75],  # Quartiles for recency segmentation
        "frequency": [25, 50, 75],
        "monetary": [25, 50, 75],
    },
    # Time window for temporal features (in hours)
    "temporal_window_hours": 24,
    # Outlier detection
    "outlier_iqr_multiplier": 1.5,  # IQR multiplier for outlier detection
    "outlier_cap_percentile": [5, 95],  # Percentiles for capping outliers
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    # Scaling method: 'standard' or 'robust'
    "scaling_method": "robust",  # RobustScaler is better for outlier-prone data
    # Feature selection
    "correlation_threshold": 0.95,  # Remove highly correlated features
    "variance_threshold": 0.01,  # Remove low variance features
    # Missing value handling
    "missing_value_strategy": "median",  # 'mean', 'median', 'mode', or 'drop'
    # Dimensionality reduction
    "use_pca": False,  # Set to True for PCA visualization
    "pca_variance_explained": 0.95,  # Keep 95% variance for PCA
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": [
        "silhouette_score",
        "davies_bouldin_score",
        "calinski_harabasz_score",
        "inertia",
    ],
    "save_plots": True,
    "plot_format": "png",
    "dpi": 300,
}

# Data loading configuration
DATA_CONFIG = {
    "chunk_size": 100000,  # Load data in chunks for memory efficiency
    "max_rows": None,  # None to load all data, or set limit for testing
    "sample_fraction": None,  # None or fraction (e.g., 0.1 for 10% sample)
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "style": "seaborn-v0_8",
    "color_palette": "Set2",
    "max_clusters_to_plot": 10,
}

# API configuration (for future deployment)
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
    "threaded": True,
}

def ensure_directories() -> None:
    """Create all necessary directories if they don't exist."""
    for dir_path in DIRECTORIES.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Also ensure models subdirectory exists
    DIRECTORIES["models"].mkdir(parents=True, exist_ok=True)


def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        "project_root": str(PROJECT_ROOT),
        "database_path": str(DATABASE_PATH),
        "venv_path": str(VENV_PATH),
        "directories": {k: str(v) for k, v in DIRECTORIES.items()},
        "model_paths": {k: str(v) for k, v in MODEL_PATHS.items()},
        "log_config": LOG_CONFIG,
        "clustering_config": CLUSTERING_CONFIG,
        "kmeans_config": KMEANS_CONFIG,
        "dbscan_config": DBSCAN_CONFIG,
        "hierarchical_config": HIERARCHICAL_CONFIG,
        "gmm_config": GMM_CONFIG,
        "spectral_config": SPECTRAL_CONFIG,
        "hdbscan_config": HDBSCAN_CONFIG,
        "stability_config": STABILITY_CONFIG,
        "feature_config": FEATURE_CONFIG,
        "preprocessing_config": PREPROCESSING_CONFIG,
        "evaluation_config": EVALUATION_CONFIG,
        "data_config": DATA_CONFIG,
        "db_table_name": DB_TABLE_NAME,
    }


if __name__ == "__main__":
    # Ensure directories exist
    ensure_directories()
    print("Configuration loaded successfully!")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Database Path: {DATABASE_PATH}")
    print(f"Database exists: {DATABASE_PATH.exists()}")

