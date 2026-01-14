"""
Feature engineering module.
Creates additional features for customer segmentation.
"""
import pandas as pd
import numpy as np
from typing import Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger("feature_engineering")


def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced behavioral features.
    
    Args:
        df: Customer-level DataFrame
    
    Returns:
        DataFrame with behavioral features added
    """
    logger.info("Creating behavioral features")
    
    df = df.copy()
    
    # Engagement intensity (events per day if time span > 0)
    if "time_span_hours" in df.columns:
        df["engagement_intensity"] = df["total_events"] / (df["time_span_hours"] / 24 + 1)  # +1 to avoid division by zero
    else:
        df["engagement_intensity"] = df["total_events"]
    
    # Purchase frequency (purchases per day)
    if "time_span_hours" in df.columns:
        df["purchase_frequency"] = df["purchase_count"] / (df["time_span_hours"] / 24 + 1)
    else:
        df["purchase_frequency"] = df["purchase_count"]
    
    # Loyalty score (combination of purchase count and frequency)
    df["loyalty_score"] = (df["purchase_count"] * 0.6) + (df["purchase_frequency"] * 0.4)
    
    # Value score (combination of spending and frequency)
    if "total_spending" in df.columns:
        df["value_score"] = (df["total_spending"] * 0.7) + (df["purchase_count"] * 0.3)
    else:
        df["value_score"] = df["purchase_count"]
    
    # Activity level categorization
    df["activity_level"] = pd.cut(
        df["total_events"],
        bins=[0, 5, 20, 100, float("inf")],
        labels=["Low", "Medium", "High", "Very High"]
    )
    
    # Spending level categorization
    if "total_spending" in df.columns:
        df["spending_level"] = pd.cut(
            df["total_spending"],
            bins=[0, 0.01, 100, 500, float("inf")],
            labels=["No Purchase", "Low", "Medium", "High"]
        )
    else:
        df["spending_level"] = "No Purchase"
    
    logger.info("Behavioral features created")
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal pattern features.
    
    Args:
        df: Customer-level DataFrame
    
    Returns:
        DataFrame with temporal features added
    """
    logger.info("Creating temporal features")
    
    df = df.copy()
    
    # Time-based engagement patterns
    if "peak_activity_hour" in df.columns:
        # Morning (6-12), Afternoon (12-18), Evening (18-24), Night (0-6)
        df["is_morning_user"] = ((df["peak_activity_hour"] >= 6) & (df["peak_activity_hour"] < 12)).astype(int)
        df["is_afternoon_user"] = ((df["peak_activity_hour"] >= 12) & (df["peak_activity_hour"] < 18)).astype(int)
        df["is_evening_user"] = ((df["peak_activity_hour"] >= 18) | (df["peak_activity_hour"] < 6)).astype(int)
    
    # Session patterns
    if "avg_session_duration_hours" in df.columns:
        # Short session (< 1 hour), Medium (1-3 hours), Long (> 3 hours)
        df["session_length_category"] = pd.cut(
            df["avg_session_duration_hours"],
            bins=[0, 1, 3, float("inf")],
            labels=["Short", "Medium", "Long"]
        )
    else:
        df["session_length_category"] = "Short"
    
    logger.info("Temporal features created")
    return df


def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engagement and diversity features.
    
    Args:
        df: Customer-level DataFrame
    
    Returns:
        DataFrame with engagement features added
    """
    logger.info("Creating engagement features")
    
    df = df.copy()
    
    # Exploration vs exploitation (diversity vs focus)
    if "product_diversity" in df.columns and "category_diversity" in df.columns:
        df["exploration_score"] = (df["product_diversity"] + df["category_diversity"]) / 2
    elif "product_diversity" in df.columns:
        df["exploration_score"] = df["product_diversity"]
    else:
        df["exploration_score"] = 0.0
    
    # Focus score (inverse of diversity)
    df["focus_score"] = 1 - df["exploration_score"] if "exploration_score" in df.columns else 0.0
    
    # Multi-category shopper indicator
    if "unique_categories" in df.columns:
        df["is_multi_category_shopper"] = (df["unique_categories"] > 3).astype(int)
        df["category_breadth"] = df["unique_categories"]
    else:
        df["is_multi_category_shopper"] = 0
        df["category_breadth"] = 0
    
    # Brand loyalty indicator
    if "unique_brands" in df.columns and "total_events" in df.columns:
        # Lower unique_brands relative to events = higher loyalty
        df["brand_loyalty_score"] = 1 - (df["unique_brands"] / (df["total_events"] + 1))
    else:
        df["brand_loyalty_score"] = 0.0
    
    logger.info("Engagement features created")
    return df


def create_price_sensitivity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create price sensitivity and preference features.
    
    Args:
        df: Customer-level DataFrame
    
    Returns:
        DataFrame with price sensitivity features added
    """
    logger.info("Creating price sensitivity features")
    
    df = df.copy()
    
    # Price sensitivity indicators
    if "price_std" in df.columns and "avg_price_viewed" in df.columns:
        # Coefficient of variation (std/mean) indicates price sensitivity
        df["price_sensitivity"] = df["price_std"] / (df["avg_price_viewed"] + 1)  # +1 to avoid division by zero
    else:
        df["price_sensitivity"] = 0.0
    
    # Price preference (prefers high vs low price items)
    if "avg_price_viewed" in df.columns:
        # Normalize by percentile
        df["price_preference_percentile"] = df["avg_price_viewed"].rank(pct=True)
        df["is_premium_shopper"] = (df["price_preference_percentile"] > 0.75).astype(int)
        df["is_budget_shopper"] = (df["price_preference_percentile"] < 0.25).astype(int)
    else:
        df["price_preference_percentile"] = 0.5
        df["is_premium_shopper"] = 0
        df["is_budget_shopper"] = 0
    
    # Price range tolerance
    if "price_range" in df.columns:
        df["price_range_tolerance"] = df["price_range"]
    else:
        df["price_range_tolerance"] = 0.0
    
    logger.info("Price sensitivity features created")
    return df


def create_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from categorical variables using Frequency Encoding.
    
    Args:
        df: Customer-level DataFrame
    
    Returns:
        DataFrame with categorical features added
    """
    logger.info("Creating categorical features")
    
    df = df.copy()
    
    # List of categorical columns to encode
    categorical_cols = ["brand", "category_code"]
    
    for col in categorical_cols:
        # Check if column exists (might be in aggregating logic)
        # Assuming aggregation might have preserved mode or list, but for now
        # let's assume we might need to join back with original data or 
        # more likely, we need to handle this at aggregation level or 
        # assume 'favorite_brand' / 'favorite_category' exists from aggregation
        
        # Checking commonly aggregated names
        target_col = f"most_frequent_{col}" if f"most_frequent_{col}" in df.columns else col
        
        if target_col in df.columns:
            # Frequency Encoding
            frequency_map = df[target_col].value_counts(normalize=True).to_dict()
            df[f"{col}_frequency_score"] = df[target_col].map(frequency_map).fillna(0)
            
            # Interaction: Brand + Category (if both exist)
            if "category_code" in categorical_cols and col == "brand":
                cat_col = f"most_frequent_category_code" if f"most_frequent_category_code" in df.columns else "category_code"
                if cat_col in df.columns:
                    df["brand_category_interaction"] = df[target_col].astype(str) + "_" + df[cat_col].astype(str)
                    interaction_freq = df["brand_category_interaction"].value_counts(normalize=True).to_dict()
                    df["brand_category_affinity"] = df["brand_category_interaction"].map(interaction_freq).fillna(0)
                    df.drop(columns=["brand_category_interaction"], inplace=True)
    
    logger.info("Categorical features created")
    return df



def engineer_features(
    df: pd.DataFrame,
    include_behavioral: bool = True,
    include_temporal: bool = True,
    include_engagement: bool = True,
    include_price: bool = True,
    include_categorical: bool = True
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Customer-level DataFrame
        include_behavioral: Whether to include behavioral features
        include_temporal: Whether to include temporal features
        include_engagement: Whether to include engagement features
        include_price: Whether to include price sensitivity features
        include_categorical: Whether to include categorical features
    
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting feature engineering pipeline")
    
    df_engineered = df.copy()
    
    if include_behavioral:
        df_engineered = create_behavioral_features(df_engineered)
    
    if include_temporal:
        df_engineered = create_temporal_features(df_engineered)
    
    if include_engagement:
        df_engineered = create_engagement_features(df_engineered)
    
    if include_price:
        df_engineered = create_price_sensitivity_features(df_engineered)
        
    if include_categorical:
        df_engineered = create_categorical_features(df_engineered)
    
    logger.info(f"Feature engineering completed. Total features: {len(df_engineered.columns)}")
    return df_engineered


def select_features_for_clustering(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Select features suitable for clustering (numeric features only).
    
    Args:
        df: DataFrame with all features
        exclude_cols: List of column names to exclude
    
    Returns:
        DataFrame with selected numeric features
    """
    logger.info("Selecting features for clustering")
    
    if exclude_cols is None:
        exclude_cols = ["user_id"]  # Default: exclude user_id
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Remove columns with infinite values
    df_selected = df[feature_cols].copy()
    df_selected = df_selected.replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Selected {len(feature_cols)} numeric features for clustering")
    logger.info(f"Features: {feature_cols[:10]}...")  # Log first 10
    
    return df_selected


if __name__ == "__main__":
    # Test feature engineering
    import pandas as pd
    from src.data.data_loader import load_and_prepare_data
    from src.data.data_aggregator import aggregate_data
    
    # Load and aggregate sample data
    df = load_and_prepare_data(max_rows=10000)
    customer_df = aggregate_data(df)
    
    # Engineer features
    engineered_df = engineer_features(customer_df)
    
    print(f"Engineered features: {len(engineered_df.columns)}")
    print(engineered_df.head())
    
    # Select features for clustering
    features_df = select_features_for_clustering(engineered_df)
    print(f"\nSelected {len(features_df.columns)} features for clustering")

