"""
Cluster evaluation module.
Provides comprehensive evaluation metrics and visualizations for clustering results.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import EVALUATION_CONFIG, DIRECTORIES
from src.utils.logger import setup_logger

logger = setup_logger("cluster_evaluator")


class ClusterEvaluator:
    """
    Evaluates clustering results and generates reports.
    """
    
    def __init__(
        self,
        model,
        preprocessor,
        save_plots: bool = True,
        plot_format: str = "png"
    ):
        """
        Initialize cluster evaluator.
        
        Args:
            model: Trained clustering model
            preprocessor: Fitted preprocessor
            save_plots: Whether to save plots
            plot_format: Plot format ('png', 'pdf', 'svg')
        """
        self.model = model
        self.preprocessor = preprocessor
        self.save_plots = save_plots
        self.plot_format = plot_format
        
        self.reports_dir = DIRECTORIES["reports"]
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        feature_names: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Evaluate clustering quality.
        
        Args:
            X: Feature matrix (preprocessed)
            labels: Cluster labels
            feature_names: Names of features
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating clustering results")
        
        n_clusters = len(np.unique(labels))
        n_noise = list(labels).count(-1) if -1 in labels else 0
        
        metrics = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "n_samples": len(labels)
        }
        
        # Calculate internal metrics
        if n_clusters > 1 and n_clusters < len(X):
            try:
                metrics["silhouette_score"] = silhouette_score(X, labels)
                logger.info(f"Silhouette score: {metrics['silhouette_score']:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate silhouette score: {str(e)}")
                metrics["silhouette_score"] = -1
            
            try:
                metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
                logger.info(f"Davies-Bouldin index: {metrics['davies_bouldin']:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate Davies-Bouldin: {str(e)}")
                metrics["davies_bouldin"] = float("inf")
            
            try:
                metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
                logger.info(f"Calinski-Harabasz index: {metrics['calinski_harabasz']:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate Calinski-Harabasz: {str(e)}")
                metrics["calinski_harabasz"] = 0
        else:
            metrics["silhouette_score"] = -1
            metrics["davies_bouldin"] = float("inf")
            metrics["calinski_harabasz"] = 0
        
        # For K-means, include inertia
        if hasattr(self.model, "inertia_"):
            metrics["inertia"] = self.model.inertia_
        
        # Cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        metrics["cluster_sizes"] = cluster_sizes
        metrics["min_cluster_size"] = min(counts) if len(counts) > 0 else 0
        metrics["max_cluster_size"] = max(counts) if len(counts) > 0 else 0
        metrics["avg_cluster_size"] = np.mean(counts) if len(counts) > 0 else 0
        
        logger.info("Evaluation completed")
        return metrics
    
    def generate_cluster_profiles(
        self,
        X_original: pd.DataFrame,
        labels: np.ndarray,
        feature_names: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Generate cluster profiles (mean feature values per cluster).
        
        Args:
            X_original: Original feature matrix (before preprocessing)
            labels: Cluster labels
            feature_names: Names of features
        
        Returns:
            DataFrame with cluster profiles
        """
        logger.info("Generating cluster profiles")
        
        if feature_names is None:
            feature_names = X_original.columns.tolist()
        
        # Create DataFrame with labels
        df_with_labels = X_original.copy()
        df_with_labels["cluster"] = labels
        
        # Calculate mean per cluster
        cluster_profiles = df_with_labels.groupby("cluster")[feature_names].mean()
        
        # Add cluster sizes
        cluster_sizes = df_with_labels["cluster"].value_counts().sort_index()
        cluster_profiles["cluster_size"] = cluster_sizes.values
        
        logger.info(f"Generated profiles for {len(cluster_profiles)} clusters")
        return cluster_profiles
    
    def plot_cluster_distribution(
        self,
        labels: np.ndarray,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot cluster size distribution.
        
        Args:
            labels: Cluster labels
            save_path: Path to save plot
        """
        logger.info("Plotting cluster distribution")
        
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(unique)), counts, color="steelblue", alpha=0.7)
        plt.xlabel("Cluster", fontsize=12)
        plt.ylabel("Number of Customers", fontsize=12)
        plt.title("Cluster Size Distribution", fontsize=14, fontweight="bold")
        plt.xticks(range(len(unique)), [f"Cluster {int(u)}" for u in unique])
        plt.grid(axis="y", alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.save_plots and save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format=self.plot_format, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_cluster_profiles(
        self,
        cluster_profiles: pd.DataFrame,
        save_path: Optional[Path] = None,
        top_n_features: int = 10
    ) -> None:
        """
        Plot cluster profiles as heatmap.
        
        Args:
            cluster_profiles: DataFrame with cluster profiles
            save_path: Path to save plot
            top_n_features: Number of top features to show
        """
        logger.info("Plotting cluster profiles")
        
        # Select top features by variance across clusters
        feature_cols = [col for col in cluster_profiles.columns if col != "cluster_size"]
        if len(feature_cols) > top_n_features:
            variances = cluster_profiles[feature_cols].var(axis=0)
            top_features = variances.nlargest(top_n_features).index.tolist()
        else:
            top_features = feature_cols
        
        # Normalize features for visualization
        plot_data = cluster_profiles[top_features].copy()
        plot_data = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min() + 1e-8)
        
        plt.figure(figsize=(12, max(6, len(top_features) * 0.5)))
        sns.heatmap(
            plot_data.T,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            cbar_kws={"label": "Normalized Feature Value"},
            xticklabels=[f"Cluster {i}" for i in plot_data.index]
        )
        plt.title("Cluster Profiles (Top Features)", fontsize=14, fontweight="bold")
        plt.xlabel("Cluster", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        
        if self.save_plots and save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format=self.plot_format, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def generate_report(
        self,
        metrics: Dict,
        cluster_profiles: Optional[pd.DataFrame] = None,
        model_info: Optional[Dict] = None
    ) -> str:
        """
        Generate markdown evaluation report.
        
        Args:
            metrics: Evaluation metrics
            cluster_profiles: Cluster profiles DataFrame
            model_info: Additional model information
        
        Returns:
            Markdown report string
        """
        logger.info("Generating evaluation report")
        
        report = "# Clustering Evaluation Report\n\n"
        
        # Model Information
        if model_info:
            report += "## Model Information\n\n"
            for key, value in model_info.items():
                report += f"- **{key}**: {value}\n"
            report += "\n"
        
        # Evaluation Metrics
        report += "## Evaluation Metrics\n\n"
        report += "### Internal Metrics\n\n"
        report += f"- **Silhouette Score**: {metrics.get('silhouette_score', 'N/A'):.4f}\n"
        report += f"- **Davies-Bouldin Index**: {metrics.get('davies_bouldin', 'N/A'):.4f}\n"
        report += f"- **Calinski-Harabasz Index**: {metrics.get('calinski_harabasz', 'N/A'):.4f}\n"
        
        if "inertia" in metrics:
            report += f"- **Inertia (Within-cluster SS)**: {metrics['inertia']:.2f}\n"
        
        report += "\n### Cluster Statistics\n\n"
        report += f"- **Number of Clusters**: {metrics.get('n_clusters', 'N/A')}\n"
        report += f"- **Number of Noise Points**: {metrics.get('n_noise', 0)}\n"
        report += f"- **Total Samples**: {metrics.get('n_samples', 'N/A')}\n"
        report += f"- **Average Cluster Size**: {metrics.get('avg_cluster_size', 'N/A'):.0f}\n"
        report += f"- **Min Cluster Size**: {metrics.get('min_cluster_size', 'N/A')}\n"
        report += f"- **Max Cluster Size**: {metrics.get('max_cluster_size', 'N/A')}\n"
        
        # Cluster Sizes
        if "cluster_sizes" in metrics:
            report += "\n### Cluster Size Distribution\n\n"
            for cluster, size in sorted(metrics["cluster_sizes"].items()):
                report += f"- **Cluster {int(cluster)}**: {size} customers\n"
        
        # Cluster Profiles
        if cluster_profiles is not None:
            report += "\n## Cluster Profiles\n\n"
            report += "### Feature Means by Cluster\n\n"
            report += cluster_profiles.to_markdown() + "\n\n"
        
        # Interpretation
        report += "\n## Interpretation\n\n"
        silhouette = metrics.get("silhouette_score", -1)
        if silhouette > 0.5:
            report += "- **Silhouette Score > 0.5**: Excellent clustering quality\n"
        elif silhouette > 0.3:
            report += "- **Silhouette Score > 0.3**: Good clustering quality\n"
        else:
            report += "- **Silhouette Score < 0.3**: Clustering quality could be improved\n"
        
        db_score = metrics.get("davies_bouldin", float("inf"))
        if db_score < 1.0:
            report += "- **Davies-Bouldin < 1.0**: Well-separated clusters\n"
        elif db_score < 2.0:
            report += "- **Davies-Bouldin < 2.0**: Reasonably separated clusters\n"
        else:
            report += "- **Davies-Bouldin > 2.0**: Clusters may overlap\n"
        
        logger.info("Report generated")
        return report
    
    def save_report(
        self,
        report: str,
        filepath: Optional[Path] = None
    ) -> None:
        """
        Save evaluation report to file.
        
        Args:
            report: Report string
            filepath: Path to save report
        """
        if filepath is None:
            filepath = self.reports_dir / "cluster_report.md"
        
        logger.info(f"Saving report to {filepath}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            f.write(report)
        
        logger.info("Report saved")


if __name__ == "__main__":
    # Test evaluator
    import pandas as pd
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from src.preprocessing.preprocessor import ClusteringPreprocessor
    
    # Create sample data
    X, _ = make_blobs(n_samples=1000, centers=4, n_features=10, random_state=42)
    X = pd.DataFrame(X)
    
    # Preprocess
    preprocessor = ClusteringPreprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    # Cluster
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_processed)
    
    # Evaluate
    evaluator = ClusterEvaluator(kmeans, preprocessor)
    metrics = evaluator.evaluate(X_processed, labels)
    
    print("Evaluation Metrics:")
    print(metrics)
    
    # Generate profiles
    profiles = evaluator.generate_cluster_profiles(X_processed, labels)
    print("\nCluster Profiles:")
    print(profiles)

