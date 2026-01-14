"""
Main training script for customer clustering.
Runs the complete pipeline: data loading, preprocessing, clustering, and evaluation.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    DATABASE_PATH,
    DB_TABLE_NAME,
    MODEL_PATHS,
    DIRECTORIES,
    ensure_directories,
    KMEANS_CONFIG,
    PREPROCESSING_CONFIG
)
from src.data.data_loader import load_and_prepare_data
from src.data.data_aggregator import aggregate_data
from src.preprocessing.feature_engineering import engineer_features, select_features_for_clustering
from src.preprocessing.preprocessor import ClusteringPreprocessor
from src.clustering.cluster_trainer import ClusteringTrainer
from src.clustering.cluster_selector import ClusterSelector
from src.evaluation.cluster_evaluator import ClusterEvaluator
from src.utils.logger import setup_logger
import joblib
import pandas as pd

logger = setup_logger("train_clusters")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting E-commerce Customer Segmentation Pipeline")
    logger.info("=" * 60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Step 1: Load data
    logger.info("\n[Step 1/7] Loading data from database...")
    try:
        df = load_and_prepare_data(
            db_path=DATABASE_PATH,
            table_name=DB_TABLE_NAME,
            max_rows=None  # Load all data
        )
        logger.info(f"Loaded {len(df):,} events")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise
    
    # Step 2: Aggregate to customer level
    logger.info("\n[Step 2/7] Aggregating to customer level...")
    try:
        customer_df = aggregate_data(df, include_rfm=True)
        logger.info(f"Aggregated to {len(customer_df):,} customers")
    except Exception as e:
        logger.error(f"Failed to aggregate data: {str(e)}")
        raise
    
    # Step 3: Feature engineering
    logger.info("\n[Step 3/7] Engineering features...")
    try:
        engineered_df = engineer_features(customer_df)
        logger.info(f"Engineered features: {len(engineered_df.columns)}")
    except Exception as e:
        logger.error(f"Failed to engineer features: {str(e)}")
        raise
    
    # Step 4: Select features for clustering
    logger.info("\n[Step 4/7] Selecting features for clustering...")
    try:
        # Include categorical frequency features
        feature_cols_to_exclude = ["user_id"]
        features_df = select_features_for_clustering(engineered_df, exclude_cols=feature_cols_to_exclude)
        logger.info(f"Selected {len(features_df.columns)} features for clustering")
    except Exception as e:
        logger.error(f"Failed to select features: {str(e)}")
        raise
    
    # Step 5: Preprocessing
    logger.info("\n[Step 5/7] Preprocessing data...")
    try:
        preprocessor = ClusteringPreprocessor(
            scaling_method=PREPROCESSING_CONFIG.get("scaling_method", "robust"),
            use_pca=PREPROCESSING_CONFIG.get("use_pca", False),
            correlation_threshold=PREPROCESSING_CONFIG.get("correlation_threshold", 0.95),
            variance_threshold=PREPROCESSING_CONFIG.get("variance_threshold", 0.01)
        )
        X_processed = preprocessor.fit_transform(features_df)
        logger.info(f"Preprocessed features: {X_processed.shape}")
        
        # Save preprocessor
        logger.info("Saving preprocessor...")
        preprocessor.save(MODEL_PATHS["preprocessor"])
        joblib.dump(preprocessor.feature_names_, MODEL_PATHS["feature_names"])
        
    except Exception as e:
        logger.error(f"Failed to preprocess data: {str(e)}")
        raise
    
    # Step 6: Determine optimal number of clusters
    logger.info("\n[Step 6/7] Determining optimal number of clusters...")
    try:
        selector = ClusterSelector(
            n_clusters_range=KMEANS_CONFIG.get("n_clusters_range", list(range(2, 11))),
            random_state=KMEANS_CONFIG.get("random_state", 42)
        )
        optimal_k = selector.select_optimal_k(X_processed, method="silhouette")
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        # Evaluate all methods for comparison
        comparison_df = selector.evaluate_all_methods(X_processed)
        comparison_path = DIRECTORIES["reports"] / "cluster_selection_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Cluster selection comparison saved to {comparison_path}")
        
    except Exception as e:
        logger.error(f"Failed to select optimal K: {str(e)}")
        raise
    
    # Step 7: Compare and train best clustering model
    logger.info("\n[Step 7/7] Comparing algorithms and training best model...")
    try:
        # Initialize trainer
        trainer = ClusteringTrainer(
            n_clusters=optimal_k,
            random_state=KMEANS_CONFIG.get("random_state", 42)
        )
        
        # Compare algorithms
        comparison_results = trainer.compare_algorithms(X_processed, n_clusters=optimal_k)
        
        # Save comparison results
        algo_comparison_path = DIRECTORIES["reports"] / "algorithm_comparison.csv"
        comparison_results.to_csv(algo_comparison_path, index=False)
        logger.info(f"Algorithm comparison saved to {algo_comparison_path}")
        
        # Determine best algorithm based on Silhouette score
        best_algo_row = comparison_results.loc[comparison_results["silhouette_score"].idxmax()]
        best_algorithm = best_algo_row["algorithm"]
        logger.info(f"Best algorithm selected: {best_algorithm}")
        
        # Train best algorithm
        # If best algorithm is a configured variant (e.g., dbscan_eps...), parse it
        # But for simplicity, we can just refit using the trainer which now holds the best metrics if we logic it right
        # The trainer.compare_algorithms doesn't persist the best model in self.model automatically in my implementation
        # So I need to fit the best one again.
        
        if "dbscan" in best_algorithm:
             # It was dbscan_eps_min... 
             # We should probably parse params or just rely on the logging
             # For now, let's just use the best algorithm string to decide
             algo_type = "dbscan"
        elif "gmm" in best_algorithm:
             algo_type = "gmm"
        elif "spectral" in best_algorithm:
             algo_type = "spectral"
        elif "hdbscan" in best_algorithm:
             algo_type = "hdbscan"
        elif "hierarchical" in best_algorithm:
             algo_type = "hierarchical"
        else:
             algo_type = "kmeans"
             
        logger.info(f"Retraining best model: {algo_type}")
        final_trainer = ClusteringTrainer(
            n_clusters=optimal_k,
            algorithm=algo_type,
            random_state=KMEANS_CONFIG.get("random_state", 42)
        )
        final_trainer.fit(X_processed)
        trainer = final_trainer # Update reference for evaluation
        
        # Save model
        logger.info("Saving trained model...")
        trainer.save(MODEL_PATHS["clusterer"])
        
    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        raise
    
    # Evaluation
    logger.info("\n[Evaluation] Evaluating clustering results...")
    try:
        evaluator = ClusterEvaluator(
            model=trainer.model,
            preprocessor=preprocessor,
            save_plots=True
        )
        
        # Evaluate
        metrics = evaluator.evaluate(X_processed, trainer.labels_)
        
        # Generate cluster profiles
        profiles = evaluator.generate_cluster_profiles(
            features_df,
            trainer.labels_,
            feature_names=features_df.columns.tolist()
        )
        
        # Generate plots
        evaluator.plot_cluster_distribution(
            trainer.labels_,
            save_path=DIRECTORIES["reports"] / "cluster_distribution.png"
        )
        evaluator.plot_cluster_profiles(
            profiles,
            save_path=DIRECTORIES["reports"] / "cluster_profiles.png"
        )
        
        # Generate and save report
        model_info = {
            "Algorithm": "K-means",
            "Number of Clusters": optimal_k,
            "Number of Features": X_processed.shape[1],
            "Number of Customers": len(customer_df),
            "Selection Method": "Silhouette Score"
        }
        
        report = evaluator.generate_report(metrics, profiles, model_info)
        evaluator.save_report(report, DIRECTORIES["reports"] / "cluster_report.md")
        
        logger.info("Evaluation completed")
        logger.info(f"Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.4f}")
        logger.info(f"Davies-Bouldin Index: {metrics.get('davies_bouldin', 'N/A'):.4f}")
        
    except Exception as e:
        logger.error(f"Failed to evaluate: {str(e)}")
        raise
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {MODEL_PATHS['clusterer']}")
    logger.info(f"Preprocessor saved to: {MODEL_PATHS['preprocessor']}")
    logger.info(f"Reports saved to: {DIRECTORIES['reports']}")


if __name__ == "__main__":
    main()

