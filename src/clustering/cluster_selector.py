"""
Cluster selection module.
Determines optimal number of clusters using multiple methods.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.utils import resample
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import KMEANS_CONFIG, STABILITY_CONFIG
from src.utils.logger import setup_logger

logger = setup_logger("cluster_selector")


class ClusterSelector:
    """
    Selects optimal number of clusters using multiple methods.
    """
    
    def __init__(
        self,
        n_clusters_range: Optional[List[int]] = None,
        random_state: int = 42,
        n_init: int = 10
    ):
        """
        Initialize cluster selector.
        
        Args:
            n_clusters_range: Range of cluster numbers to test
            random_state: Random state for reproducibility
            n_init: Number of initializations for K-means
        """
        self.n_clusters_range = n_clusters_range or KMEANS_CONFIG.get("n_clusters_range", list(range(2, 11)))
        self.random_state = random_state
        self.n_init = n_init
        
        self.scores_ = {}
        self.optimal_k_ = None
    
    def elbow_method(self, X: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate inertia (within-cluster sum of squares) for different K values.
        Lower inertia is better, but we look for the "elbow" point.
        
        Args:
            X: Feature matrix
        
        Returns:
            Dictionary mapping K to inertia
        """
        logger.info("Calculating elbow method scores")
        inertias = {}
        
        for k in self.n_clusters_range:
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=self.n_init,
                random_state=self.random_state,
                n_jobs=-1
            )
            kmeans.fit(X)
            inertias[k] = kmeans.inertia_
            logger.debug(f"K={k}: Inertia={inertias[k]:.2f}")
        
        logger.info("Elbow method completed")
        return inertias
    
    def silhouette_method(self, X: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate silhouette scores for different K values.
        Higher silhouette score indicates better separation.
        
        Args:
            X: Feature matrix
        
        Returns:
            Dictionary mapping K to silhouette score
        """
        logger.info("Calculating silhouette scores")
        silhouette_scores = {}
        
        for k in self.n_clusters_range:
            if k >= len(X):
                logger.warning(f"K={k} is >= number of samples, skipping")
                continue
            
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=self.n_init,
                random_state=self.random_state,
                n_jobs=-1
            )
            labels = kmeans.fit_predict(X)
            
            if len(np.unique(labels)) < 2:
                logger.warning(f"K={k} resulted in < 2 clusters, skipping silhouette")
                silhouette_scores[k] = -1
            else:
                score = silhouette_score(X, labels)
                silhouette_scores[k] = score
                logger.debug(f"K={k}: Silhouette={score:.4f}")
        
        logger.info("Silhouette method completed")
        return silhouette_scores
    
    def davies_bouldin_method(self, X: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate Davies-Bouldin index for different K values.
        Lower score indicates better clustering.
        
        Args:
            X: Feature matrix
        
        Returns:
            Dictionary mapping K to Davies-Bouldin score
        """
        logger.info("Calculating Davies-Bouldin scores")
        db_scores = {}
        
        for k in self.n_clusters_range:
            if k >= len(X):
                continue
            
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=self.n_init,
                random_state=self.random_state,
                n_jobs=-1
            )
            labels = kmeans.fit_predict(X)
            
            if len(np.unique(labels)) < 2:
                db_scores[k] = float("inf")
            else:
                score = davies_bouldin_score(X, labels)
                db_scores[k] = score
                logger.debug(f"K={k}: Davies-Bouldin={score:.4f}")
        
        logger.info("Davies-Bouldin method completed")
        return db_scores
    
    def calinski_harabasz_method(self, X: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate Calinski-Harabasz index for different K values.
        Higher score indicates better clustering.
        
        Args:
            X: Feature matrix
        
        Returns:
            Dictionary mapping K to Calinski-Harabasz score
        """
        logger.info("Calculating Calinski-Harabasz scores")
        ch_scores = {}
        
        for k in self.n_clusters_range:
            if k >= len(X):
                continue
            
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=self.n_init,
                random_state=self.random_state,
                n_jobs=-1
            )
            labels = kmeans.fit_predict(X)
            
            if len(np.unique(labels)) < 2:
                ch_scores[k] = 0
            else:
                score = calinski_harabasz_score(X, labels)
                ch_scores[k] = score
                logger.debug(f"K={k}: Calinski-Harabasz={score:.4f}")
        
        logger.info("Calinski-Harabasz method completed")
        return ch_scores
    
    def stability_method(self, X: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate stability score (Adjusted Rand Index) for different K values using bootstrapping.
        Higher score indicates more stable clustering.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary mapping K to stability score
        """
        logger.info("Calculating stability scores")
        stability_scores = {}
        
        n_bootstrap = STABILITY_CONFIG.get("n_bootstrap", 20)
        sample_fraction = STABILITY_CONFIG.get("sample_fraction", 0.8)
        
        for k in self.n_clusters_range:
            if k >= len(X):
                continue
                
            scores = []
            for _ in range(n_bootstrap):
                # Bootstrap sample
                X_sample = resample(X, n_samples=int(len(X) * sample_fraction), random_state=None) # Random state None for variation
                
                # Cluster the sample
                kmeans_sample = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init=1, # Speed up
                    random_state=None
                )
                labels_sample = kmeans_sample.fit_predict(X_sample)
                
                # Predict on sample using model trained on full data (or retraining on full data and subsetting)
                # To measure stability properly: 
                # 1. Cluster full data -> labels_full
                # 2. Subset labels_full to indices of X_sample -> labels_full_subset
                # 3. Compare labels_sample vs labels_full_subset
                
                # Train on full data (we can cache this if needed, but for now simple loop)
                # Ideally we pass in the full model or cache it, but let's just retrain for simplicity 
                # inside the loop might be slow. 
                # Better: Train K-Means on full data ONCE per K.
                pass
            
            # Optimization: Move full training outside bootstrap loop
            kmeans_full = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=1,
                random_state=self.random_state
            )
            labels_full_all = kmeans_full.fit_predict(X)
            labels_full_series = pd.Series(labels_full_all, index=X.index)
            
            bootstrap_scores = []
            for i in range(n_bootstrap):
                # Bootstrap indices
                indices = resample(X.index, n_samples=int(len(X) * sample_fraction), random_state=None)
                X_subset = X.loc[indices]
                
                # Cluster subset
                kmeans_subset = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init=1, 
                    random_state=None
                )
                labels_subset = kmeans_subset.fit_predict(X_subset)
                
                # Get labels from full model for these indices
                labels_full_subset = labels_full_series.loc[indices].values
                
                # Compare
                ari = adjusted_rand_score(labels_full_subset, labels_subset)
                bootstrap_scores.append(ari)
            
            avg_score = np.mean(bootstrap_scores)
            stability_scores[k] = avg_score
            logger.debug(f"K={k}: Stability={avg_score:.4f}")
            
        logger.info("Stability method completed")
        return stability_scores

    def select_optimal_k(self, X: pd.DataFrame, method: str = "silhouette") -> int:
        """
        Select optimal number of clusters using specified method.
        
        Args:
            X: Feature matrix
            method: Method to use ('silhouette', 'davies_bouldin', 'calinski_harabasz', 'elbow')
        
        Returns:
            Optimal number of clusters
        """
        logger.info(f"Selecting optimal K using {method} method")
        
        if method == "silhouette":
            scores = self.silhouette_method(X)
            optimal_k = max(scores.items(), key=lambda x: x[1])[0]
        elif method == "davies_bouldin":
            scores = self.davies_bouldin_method(X)
            optimal_k = min(scores.items(), key=lambda x: x[1])[0]
        elif method == "calinski_harabasz":
            scores = self.calinski_harabasz_method(X)
            optimal_k = max(scores.items(), key=lambda x: x[1])[0]
        elif method == "stability":
            scores = self.stability_method(X)
            optimal_k = max(scores.items(), key=lambda x: x[1])[0]
        elif method == "elbow":
            scores = self.elbow_method(X)
            # Elbow method requires finding the "elbow" point
            # Simple approach: find K where decrease in inertia slows down
            k_values = sorted(scores.keys())
            inertias = [scores[k] for k in k_values]
            if len(inertias) >= 3:
                # Calculate rate of change
                decreases = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
                # Find where decrease rate drops significantly
                if len(decreases) >= 2:
                    decrease_rates = [decreases[i] / decreases[i+1] if decreases[i+1] > 0 else 0 
                                     for i in range(len(decreases)-1)]
                    # Find first significant drop
                    optimal_k = k_values[1]  # Default
                    for i, rate in enumerate(decrease_rates):
                        if rate < 0.5:  # Significant drop
                            optimal_k = k_values[i+1]
                            break
                else:
                    optimal_k = k_values[0]
            else:
                optimal_k = k_values[0]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.optimal_k_ = optimal_k
        self.scores_[method] = scores
        
        logger.info(f"Optimal K selected: {optimal_k}")
        return optimal_k
    
    def evaluate_all_methods(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate all methods and return comparison DataFrame.
        
        Args:
            X: Feature matrix
        
        Returns:
            DataFrame with scores for all methods
        """
        logger.info("Evaluating all cluster selection methods")
        
        # Calculate all scores
        elbow_scores = self.elbow_method(X)
        silhouette_scores = self.silhouette_method(X)
        db_scores = self.davies_bouldin_method(X)
        ch_scores = self.calinski_harabasz_method(X)
        # Stability can be slow, calculate only if dataset is small enough or explicitly requested
        if len(X) < 5000: # Limit for performance
            stability_scores = self.stability_method(X)
        else:
            stability_scores = {}
        
        # Create comparison DataFrame
        results = []
        for k in self.n_clusters_range:
            if k in elbow_scores:
                results.append({
                    "n_clusters": k,
                    "inertia": elbow_scores[k],
                    "silhouette_score": silhouette_scores.get(k, np.nan),
                    "davies_bouldin": db_scores.get(k, np.nan),
                    "calinski_harabasz": ch_scores.get(k, np.nan),
                    "stability": stability_scores.get(k, np.nan)
                })
        
        results_df = pd.DataFrame(results)
        
        # Store scores
        self.scores_ = {
            "elbow": elbow_scores,
            "silhouette": silhouette_scores,
            "davies_bouldin": db_scores,
            "calinski_harabasz": ch_scores,
            "stability": stability_scores if 'stability_scores' in locals() else {}
        }
        
        logger.info("All methods evaluated")
        return results_df


if __name__ == "__main__":
    # Test cluster selector
    import pandas as pd
    from sklearn.datasets import make_blobs
    
    # Create sample data
    X, _ = make_blobs(n_samples=1000, centers=4, n_features=10, random_state=42)
    X = pd.DataFrame(X)
    
    # Test selector
    selector = ClusterSelector(n_clusters_range=list(range(2, 8)))
    results = selector.evaluate_all_methods(X)
    
    print("Cluster Selection Results:")
    print(results)
    
    optimal_k = selector.select_optimal_k(X, method="silhouette")
    print(f"\nOptimal K (silhouette): {optimal_k}")

