"""
Clustering model trainer module.
Implements multiple clustering algorithms and selects the best one.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    # Fallback for older sklearn versions
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        HDBSCAN = None
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Optional, Tuple, List, Any
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import (
    KMEANS_CONFIG, DBSCAN_CONFIG, HIERARCHICAL_CONFIG, CLUSTERING_CONFIG,
    GMM_CONFIG, SPECTRAL_CONFIG, HDBSCAN_CONFIG
)
from src.clustering.cluster_selector import ClusterSelector
from src.utils.logger import setup_logger

logger = setup_logger("cluster_trainer")


class ClusteringTrainer:
    """
    Trains multiple clustering algorithms and selects the best one.
    """
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        algorithm: str = "kmeans",
        random_state: int = 42
    ):
        """
        Initialize clustering trainer.
        
        Args:
            n_clusters: Number of clusters (if None, will be determined automatically)
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical', 'gmm', 'spectral', 'hdbscan')
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.random_state = random_state
        
        self.model = None
        self.labels_ = None
        self.metrics_ = {}
        self.best_params_ = {}
    
    def train_kmeans(
        self,
        X: pd.DataFrame,
        n_clusters: Optional[int] = None,
        n_init: int = 10,
        max_iter: int = 300
    ) -> Tuple[KMeans, np.ndarray]:
        """
        Train K-means clustering model.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            n_init: Number of initializations
            max_iter: Maximum iterations
        
        Returns:
            Tuple of (trained model, cluster labels)
        """
        logger.info(f"Training K-means with {n_clusters} clusters")
        
        if n_clusters is None:
            n_clusters = self.n_clusters or 5
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        labels = kmeans.fit_predict(X)
        
        logger.info(f"K-means training completed. Clusters: {len(np.unique(labels))}")
        return kmeans, labels
    
    def train_dbscan(
        self,
        X: pd.DataFrame,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> Tuple[DBSCAN, np.ndarray]:
        """
        Train DBSCAN clustering model.
        
        Args:
            X: Feature matrix
            eps: Maximum distance between samples in same neighborhood
            min_samples: Minimum samples in neighborhood
        
        Returns:
            Tuple of (trained model, cluster labels)
        """
        logger.info(f"Training DBSCAN with eps={eps}, min_samples={min_samples}")
        
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean",
            n_jobs=-1
        )
        
        labels = dbscan.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"DBSCAN training completed. Clusters: {n_clusters}, Noise points: {n_noise}")
        return dbscan, labels
    
    def train_hierarchical(
        self,
        X: pd.DataFrame,
        n_clusters: Optional[int] = None,
        linkage: str = "ward"
    ) -> Tuple[AgglomerativeClustering, np.ndarray]:
        """
        Train Hierarchical (Agglomerative) clustering model.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average')
        
        Returns:
            Tuple of (trained model, cluster labels)
        """
        logger.info(f"Training Hierarchical clustering with {n_clusters} clusters, linkage={linkage}")
        
        if n_clusters is None:
            n_clusters = self.n_clusters or 5
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric="euclidean" if linkage != "ward" else None
        )
        
        labels = hierarchical.fit_predict(X)
        
        logger.info(f"Hierarchical clustering completed. Clusters: {len(np.unique(labels))}")
        return hierarchical, labels
    
    def train_gmm(
        self,
        X: pd.DataFrame,
        n_components: Optional[int] = None,
        covariance_type: str = "full"
    ) -> Tuple[GaussianMixture, np.ndarray]:
        """
        Train Gaussian Mixture Model.
        
        Args:
            X: Feature matrix
            n_components: Number of mixture components
            covariance_type: Type of covariance parameters
        
        Returns:
            Tuple of (trained model, cluster labels)
        """
        logger.info(f"Training GMM with {n_components} components, covariance={covariance_type}")
        
        if n_components is None:
            n_components = self.n_clusters or 5
            
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=GMM_CONFIG.get("n_init", 10),
            random_state=self.random_state
        )
        
        labels = gmm.fit_predict(X)
        
        logger.info(f"GMM training completed. Clusters: {len(np.unique(labels))}")
        return gmm, labels

    def train_spectral(
        self,
        X: pd.DataFrame,
        n_clusters: Optional[int] = None,
        affinity: str = "rbf"
    ) -> Tuple[SpectralClustering, np.ndarray]:
        """
        Train Spectral Clustering model.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            affinity: Affinity kernel
        
        Returns:
            Tuple of (trained model, cluster labels)
        """
        logger.info(f"Training Spectral Clustering with {n_clusters} clusters")
        
        if n_clusters is None:
            n_clusters = self.n_clusters or 5
            
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            n_neighbors=SPECTRAL_CONFIG.get("n_neighbors", 10),
            assign_labels=SPECTRAL_CONFIG.get("assign_labels", "kmeans"),
            random_state=self.random_state,
            n_jobs=-1
        )
        
        labels = spectral.fit_predict(X)
        self.model = spectral # Spectral doesn't have a predict method for new data usually
        
        logger.info(f"Spectral Clustering completed. Clusters: {len(np.unique(labels))}")
        return spectral, labels

    def train_hdbscan(
        self,
        X: pd.DataFrame,
        min_cluster_size: int = 5,
        min_samples: int = 5
    ) -> Tuple[Any, np.ndarray]:
        """
        Train HDBSCAN clustering model.
        
        Args:
            X: Feature matrix
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in neighborhood
        
        Returns:
            Tuple of (trained model, cluster labels)
        """
        if HDBSCAN is None:
            raise ImportError("HDBSCAN is not available. Please install scikit-learn >= 1.3.0 or hdbscan package.")

        logger.info(f"Training HDBSCAN with min_cluster_size={min_cluster_size}")
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=HDBSCAN_CONFIG.get("metric", "euclidean"),
            cluster_selection_epsilon=HDBSCAN_CONFIG.get("cluster_selection_epsilon", 0.0),
            n_jobs=-1
        )
        
        labels = hdbscan_model.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"HDBSCAN training completed. Clusters: {n_clusters}, Noise points: {n_noise}")
        return hdbscan_model, labels
    
    def evaluate_clustering(
        self,
        X: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
        
        Returns:
            Dictionary of evaluation metrics
        """
        n_clusters = len(np.unique(labels))
        n_noise = list(labels).count(-1) if -1 in labels else 0
        
        metrics = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "n_samples": len(labels)
        }
        
        # Calculate metrics only if we have valid clusters
        if n_clusters > 1 and n_clusters < len(X):
            try:
                metrics["silhouette_score"] = silhouette_score(X, labels)
            except:
                metrics["silhouette_score"] = -1
            
            try:
                metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
            except:
                metrics["davies_bouldin"] = float("inf")
            
            try:
                metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
            except:
                metrics["calinski_harabasz"] = 0
        else:
            metrics["silhouette_score"] = -1
            metrics["davies_bouldin"] = float("inf")
            metrics["calinski_harabasz"] = 0
        
        # For K-means, also calculate inertia
        if hasattr(self.model, "inertia_"):
            metrics["inertia"] = self.model.inertia_
        
        return metrics
    
    def fit(
        self,
        X: pd.DataFrame,
        auto_select_k: bool = False,
        method: str = "silhouette"
    ) -> 'ClusteringTrainer':
        """
        Fit clustering model.
        
        Args:
            X: Feature matrix
            auto_select_k: Whether to automatically select optimal K
            method: Method for selecting K ('silhouette', 'davies_bouldin', 'calinski_harabasz')
        
        Returns:
            Self
        """
        logger.info(f"Training {self.algorithm} clustering model")
        
        # Auto-select K if needed
        if auto_select_k and self.n_clusters is None:
            logger.info("Auto-selecting optimal number of clusters")
            selector = ClusterSelector(random_state=self.random_state)
            self.n_clusters = selector.select_optimal_k(X, method=method)
            self.best_params_["n_clusters"] = self.n_clusters
            self.best_params_["selection_method"] = method
        
        # Train based on algorithm
        if self.algorithm == "kmeans":
            self.model, self.labels_ = self.train_kmeans(
                X,
                n_clusters=self.n_clusters
            )
        elif self.algorithm == "dbscan":
            # Try default parameters first
            eps = DBSCAN_CONFIG.get("eps_range", [0.5, 1.0, 1.5])[0]
            min_samples = DBSCAN_CONFIG.get("min_samples_range", [5, 10])[0]
            self.model, self.labels_ = self.train_dbscan(X, eps=eps, min_samples=min_samples)
            self.best_params_["eps"] = eps
            self.best_params_["min_samples"] = min_samples
        elif self.algorithm == "hierarchical":
            linkage = HIERARCHICAL_CONFIG.get("linkage_options", ["ward"])[0]
            self.model, self.labels_ = self.train_hierarchical(
                X,
                n_clusters=self.n_clusters,
                linkage=linkage
            )
            self.best_params_["linkage"] = linkage
        elif self.algorithm == "gmm":
            cov_type = GMM_CONFIG.get("covariance_type", "full")
            self.model, self.labels_ = self.train_gmm(
                X,
                n_components=self.n_clusters,
                covariance_type=cov_type
            )
            self.best_params_["covariance_type"] = cov_type
        elif self.algorithm == "spectral":
            affinity = SPECTRAL_CONFIG.get("affinity", "rbf")
            self.model, self.labels_ = self.train_spectral(
                X,
                n_clusters=self.n_clusters,
                affinity=affinity
            )
            self.best_params_["affinity"] = affinity
        elif self.algorithm == "hdbscan":
            min_cluster_size = HDBSCAN_CONFIG.get("min_cluster_size_range", [5])[0]
            min_samples = HDBSCAN_CONFIG.get("min_samples_range", [None])[0]
            self.model, self.labels_ = self.train_hdbscan(
                X,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples if min_samples is not None else min_cluster_size
            )
            self.best_params_["min_cluster_size"] = min_cluster_size
            self.best_params_["min_samples"] = min_samples
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Evaluate
        self.metrics_ = self.evaluate_clustering(X, self.labels_)
        
        logger.info(f"Training completed. Metrics: {self.metrics_}")
        return self
    
    def compare_algorithms(
        self,
        X: pd.DataFrame,
        n_clusters: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compare multiple clustering algorithms.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters for K-means and Hierarchical
        
        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing multiple clustering algorithms")
        
        results = []
        
        # Test K-means
        logger.info("Testing K-means...")
        kmeans_trainer = ClusteringTrainer(
            n_clusters=n_clusters,
            algorithm="kmeans",
            random_state=self.random_state
        )
        kmeans_trainer.fit(X, auto_select_k=(n_clusters is None))
        results.append({
            "algorithm": "kmeans",
            **kmeans_trainer.metrics_
        })
        
        # Test Hierarchical
        logger.info("Testing Hierarchical clustering...")
        hierarchical_trainer = ClusteringTrainer(
            n_clusters=n_clusters or kmeans_trainer.n_clusters,
            algorithm="hierarchical",
            random_state=self.random_state
        )
        hierarchical_trainer.fit(X)
        results.append({
            "algorithm": "hierarchical",
            **hierarchical_trainer.metrics_
        })
        
        # Test DBSCAN (multiple parameter combinations)
        logger.info("Testing DBSCAN...")
        eps_range = DBSCAN_CONFIG.get("eps_range", [0.5, 1.0, 1.5])
        min_samples_range = DBSCAN_CONFIG.get("min_samples_range", [5, 10])
        
        for eps in eps_range[:2]:  # Test first 2 eps values
            for min_samples in min_samples_range[:2]:  # Test first 2 min_samples
                dbscan_trainer = ClusteringTrainer(
                    algorithm="dbscan",
                    random_state=self.random_state
                )
                dbscan_trainer.model, dbscan_trainer.labels_ = dbscan_trainer.train_dbscan(
                    X, eps=eps, min_samples=min_samples
                )
                dbscan_trainer.metrics_ = dbscan_trainer.evaluate_clustering(X, dbscan_trainer.labels_)
                results.append({
                    "algorithm": f"dbscan_eps{eps}_min{min_samples}",
                    **dbscan_trainer.metrics_
                })
        
        # Test GMM
        logger.info("Testing GMM...")
        gmm_trainer = ClusteringTrainer(
            n_clusters=n_clusters or kmeans_trainer.n_clusters,
            algorithm="gmm",
            random_state=self.random_state
        )
        gmm_trainer.fit(X)
        results.append({
            "algorithm": "gmm",
            **gmm_trainer.metrics_
        })

        # Test Spectral
        # Spectral can be slow, so we might want to skip for very large datasets
        if len(X) < 10000: # Simple guard
            logger.info("Testing Spectral Clustering...")
            spectral_trainer = ClusteringTrainer(
                n_clusters=n_clusters or kmeans_trainer.n_clusters,
                algorithm="spectral",
                random_state=self.random_state
            )
            spectral_trainer.fit(X)
            results.append({
                "algorithm": "spectral",
                **spectral_trainer.metrics_
            })
        
        # Test HDBSCAN
        if HDBSCAN is not None:
            logger.info("Testing HDBSCAN...")
            hdbscan_trainer = ClusteringTrainer(
                algorithm="hdbscan",
                random_state=self.random_state
            )
            hdbscan_trainer.fit(X)
            results.append({
                "algorithm": "hdbscan",
                **hdbscan_trainer.metrics_
            })
        
        comparison_df = pd.DataFrame(results)
        
        # Select best algorithm based on silhouette score
        valid_results = comparison_df[comparison_df["silhouette_score"] > -1]
        if len(valid_results) > 0:
            best_idx = valid_results["silhouette_score"].idxmax()
            best_algorithm = comparison_df.loc[best_idx, "algorithm"]
            logger.info(f"Best algorithm (by silhouette): {best_algorithm}")
        
        logger.info("Algorithm comparison completed")
        return comparison_df
    
    def save(self, filepath: Path) -> None:
        """Save trained model to disk."""
        logger.info(f"Saving model to {filepath}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Model saved")
    
    @staticmethod
    def load(filepath: Path) -> 'ClusteringTrainer':
        """Load trained model from disk."""
        logger.info(f"Loading model from {filepath}")
        return joblib.load(filepath)


if __name__ == "__main__":
    # Test cluster trainer
    import pandas as pd
    from sklearn.datasets import make_blobs
    
    # Create sample data
    X, _ = make_blobs(n_samples=1000, centers=4, n_features=10, random_state=42)
    X = pd.DataFrame(X)
    
    # Test trainer
    trainer = ClusteringTrainer(n_clusters=4, algorithm="kmeans")
    trainer.fit(X)
    
    print(f"Labels: {np.unique(trainer.labels_, return_counts=True)}")
    print(f"Metrics: {trainer.metrics_}")
    
    # Compare algorithms
    comparison = trainer.compare_algorithms(X, n_clusters=4)
    print("\nAlgorithm Comparison:")
    print(comparison)

