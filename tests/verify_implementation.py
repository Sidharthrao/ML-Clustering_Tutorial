
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.feature_engineering import create_categorical_features
from src.clustering.cluster_trainer import ClusteringTrainer
from src.clustering.cluster_selector import ClusterSelector
from config.config import HDBSCAN_CONFIG

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verification")

def verify_categorical_features():
    logger.info("Verifying categorical features...")
    df = pd.DataFrame({
        "brand": ["A", "B", "A", "C", "B", "A"],
        "category_code": ["X", "Y", "X", "Z", "Y", "X"],
        "price": [10, 20, 10, 30, 20, 10]
    })
    
    df_new = create_categorical_features(df)
    
    assert "brand_frequency_score" in df_new.columns
    assert "brand_category_affinity" in df_new.columns
    logger.info("Categorical features verified.")

def verify_models():
    logger.info("Verifying new clustering models...")
    # Create synthetic numeric data
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    
    # Test GMM
    trainer = ClusteringTrainer(n_clusters=3, algorithm="gmm")
    trainer.fit(X)
    assert len(np.unique(trainer.labels_)) > 1
    logger.info("GMM verified.")
    
    # Test Spectral
    trainer = ClusteringTrainer(n_clusters=3, algorithm="spectral")
    trainer.fit(X)
    assert len(np.unique(trainer.labels_)) > 1
    logger.info("Spectral verified.")
    
    # Test HDBSCAN
    # Use config default
    trainer = ClusteringTrainer(algorithm="hdbscan")
    try:
        trainer.fit(X)
        logger.info(f"HDBSCAN verified. Clusters: {len(np.unique(trainer.labels_))}")
    except ImportError:
        logger.warning("HDBSCAN not available.")
    except Exception as e:
        logger.error(f"HDBSCAN failed: {e}")
        raise

def verify_stability():
    logger.info("Verifying stability analysis...")
    X = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
    
    selector = ClusterSelector(n_clusters_range=[2, 3])
    scores = selector.stability_method(X)
    
    assert 2 in scores
    assert 3 in scores
    logger.info(f"Stability scores: {scores}")
    logger.info("Stability analysis verified.")

if __name__ == "__main__":
    verify_categorical_features()
    verify_models()
    verify_stability()
    print("ALL VERIFICATIONS PASSED")
