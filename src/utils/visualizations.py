"""
Visualization utility functions for clustering analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import DIRECTORIES, VISUALIZATION_CONFIG
from src.utils.logger import setup_logger

logger = setup_logger("visualizations")

# Set style
plt.style.use(VISUALIZATION_CONFIG.get("style", "seaborn-v0_8"))
sns.set_palette(VISUALIZATION_CONFIG.get("color_palette", "Set2"))


def plot_pca_clusters(
    X: pd.DataFrame,
    labels: np.ndarray,
    n_components: int = 2,
    save_path: Optional[Path] = None,
    title: str = "Clusters in PCA Space"
) -> None:
    """
    Plot clusters in PCA-reduced space.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        n_components: Number of PCA components (2 or 3)
        save_path: Path to save plot
        title: Plot title
    """
    logger.info(f"Plotting clusters in {n_components}D PCA space")
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Create figure
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=labels, cmap="tab10", alpha=0.6, s=50
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.grid(alpha=0.3)
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
            c=labels, cmap="tab10", alpha=0.6, s=50
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=10)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=10)
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)", fontsize=10)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.colorbar(scatter, ax=ax, label="Cluster")
    else:
        raise ValueError("n_components must be 2 or 3")
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"PCA plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_cluster_comparison(
    cluster_profiles: pd.DataFrame,
    feature_names: List[str],
    save_path: Optional[Path] = None
) -> None:
    """
    Plot radar chart comparing clusters across features.
    
    Args:
        cluster_profiles: DataFrame with cluster profiles
        feature_names: Names of features to compare
        save_path: Path to save plot
    """
    logger.info("Plotting cluster comparison radar chart")
    
    # Select top features
    n_features = min(len(feature_names), 8)  # Limit to 8 for readability
    top_features = feature_names[:n_features]
    
    # Normalize features
    plot_data = cluster_profiles[top_features].copy()
    plot_data = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min() + 1e-8)
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for cluster_idx in plot_data.index:
        values = plot_data.loc[cluster_idx].tolist()
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_idx}')
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_features)
    ax.set_ylim(0, 1)
    ax.set_title('Cluster Comparison', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Radar chart saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_distributions_by_cluster(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_names: List[str],
    n_features: int = 6,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot feature distributions by cluster.
    
    Args:
        df: Feature DataFrame
        labels: Cluster labels
        feature_names: Names of features to plot
        n_features: Number of features to plot
        save_path: Path to save plot
    """
    logger.info("Plotting feature distributions by cluster")
    
    n_features = min(n_features, len(feature_names))
    top_features = feature_names[:n_features]
    
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    df_with_labels = df.copy()
    df_with_labels["cluster"] = labels
    
    for i, feature in enumerate(top_features):
        if feature in df.columns:
            ax = axes[i]
            for cluster in sorted(df_with_labels["cluster"].unique()):
                cluster_data = df_with_labels[df_with_labels["cluster"] == cluster][feature]
                ax.hist(cluster_data, alpha=0.6, label=f"Cluster {cluster}", bins=30)
            
            ax.set_title(f"{feature} Distribution", fontsize=12, fontweight="bold")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(alpha=0.3)
    
    # Hide extra subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Feature distributions plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "Feature Correlation Matrix"
) -> None:
    """
    Plot correlation matrix.
    
    Args:
        df: DataFrame with numeric features
        save_path: Path to save plot
        title: Plot title
    """
    logger.info("Plotting correlation matrix")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(12, 10))
    correlation = numeric_df.corr()
    
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(
        correlation,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Correlation matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_elbow_curve(
    n_clusters_range: List[int],
    inertias: List[float],
    save_path: Optional[Path] = None
) -> None:
    """
    Plot elbow curve for K-means.
    
    Args:
        n_clusters_range: Range of cluster numbers
        inertias: Inertia values for each K
        save_path: Path to save plot
    """
    logger.info("Plotting elbow curve")
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Inertia (Within-cluster SS)", fontsize=12)
    plt.title("Elbow Method for Optimal K", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Elbow curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_silhouette_scores(
    n_clusters_range: List[int],
    silhouette_scores: List[float],
    save_path: Optional[Path] = None
) -> None:
    """
    Plot silhouette scores for different K values.
    
    Args:
        n_clusters_range: Range of cluster numbers
        silhouette_scores: Silhouette scores for each K
        save_path: Path to save plot
    """
    logger.info("Plotting silhouette scores")
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.title("Silhouette Score for Different K Values", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Silhouette scores plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test visualizations
    import pandas as pd
    from sklearn.datasets import make_blobs
    
    # Create sample data
    X, labels = make_blobs(n_samples=1000, centers=4, n_features=10, random_state=42)
    X = pd.DataFrame(X)
    
    # Test PCA plot
    plot_pca_clusters(X, labels, save_path=DIRECTORIES["reports"] / "test_pca.png")
    
    # Test correlation matrix
    plot_correlation_matrix(X, save_path=DIRECTORIES["reports"] / "test_correlation.png")

