"""
Data aggregation module.
Aggregates event-level data to customer-level features for clustering.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger("data_aggregator")


def aggregate_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate event-level data to customer-level features.
    
    This function creates customer-level features including:
    - Event counts (views, carts, purchases)
    - Spending metrics (total, average)
    - Engagement metrics (unique products, categories, sessions)
    - Temporal features (first/last event, session duration)
    - Behavioral metrics (conversion rates, diversity)
    
    Args:
        df: Event-level DataFrame with columns: event_time, event_type, product_id,
            category_id, category_code, brand, price, user_id, user_session
    
    Returns:
        DataFrame with one row per customer and aggregated features
    """
    logger.info("Aggregating customer-level features")
    logger.info(f"Input data shape: {df.shape}")
    
    # Ensure event_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["event_time"]):
        df["event_time"] = pd.to_datetime(df["event_time"])
    
    # Group by user_id
    customer_features = []
    
    for user_id, user_data in df.groupby("user_id"):
        features = {"user_id": user_id}
        
        # === EVENT COUNTS ===
        event_counts = user_data["event_type"].value_counts()
        features["total_events"] = len(user_data)
        features["view_count"] = event_counts.get("view", 0)
        features["cart_count"] = event_counts.get("cart", 0)
        features["purchase_count"] = event_counts.get("purchase", 0)
        
        # === SPENDING METRICS ===
        purchase_data = user_data[user_data["event_type"] == "purchase"]
        if len(purchase_data) > 0:
            features["total_spending"] = purchase_data["price"].sum()
            features["avg_order_value"] = purchase_data["price"].mean()
            features["max_order_value"] = purchase_data["price"].max()
            features["min_order_value"] = purchase_data["price"].min()
        else:
            features["total_spending"] = 0.0
            features["avg_order_value"] = 0.0
            features["max_order_value"] = 0.0
            features["min_order_value"] = 0.0
        
        # Average price of products viewed (even if not purchased)
        viewed_data = user_data[user_data["event_type"] == "view"]
        if len(viewed_data) > 0:
            features["avg_price_viewed"] = viewed_data["price"].mean()
        else:
            features["avg_price_viewed"] = 0.0
        
        # === UNIQUE COUNTS ===
        features["unique_products_viewed"] = user_data["product_id"].nunique()
        features["unique_products_purchased"] = purchase_data["product_id"].nunique() if len(purchase_data) > 0 else 0
        features["unique_categories"] = user_data["category_id"].nunique()
        features["unique_brands"] = user_data["brand"].dropna().nunique() if "brand" in user_data.columns else 0
        features["unique_sessions"] = user_data["user_session"].nunique()
        
        # === TEMPORAL FEATURES ===
        features["first_event_time"] = user_data["event_time"].min()
        features["last_event_time"] = user_data["event_time"].max()
        features["time_span_hours"] = (features["last_event_time"] - features["first_event_time"]).total_seconds() / 3600
        features["time_span_hours"] = max(features["time_span_hours"], 0)  # Handle same timestamp
        
        # Average time between events (in hours)
        if len(user_data) > 1:
            user_data_sorted = user_data.sort_values("event_time")
            time_diffs = user_data_sorted["event_time"].diff().dt.total_seconds() / 3600
            features["avg_time_between_events_hours"] = time_diffs[time_diffs > 0].mean() if len(time_diffs[time_diffs > 0]) > 0 else 0
        else:
            features["avg_time_between_events_hours"] = 0.0
        
        # Peak activity hour (hour with most events)
        user_data["hour"] = user_data["event_time"].dt.hour
        if len(user_data) > 0:
            features["peak_activity_hour"] = user_data["hour"].mode()[0] if len(user_data["hour"].mode()) > 0 else user_data["hour"].median()
        else:
            features["peak_activity_hour"] = 12  # Default to noon
        
        # === BEHAVIORAL METRICS ===
        # Conversion rates
        if features["view_count"] > 0:
            features["purchase_conversion_rate"] = features["purchase_count"] / features["view_count"]
            features["cart_conversion_rate"] = features["cart_count"] / features["view_count"]
        else:
            features["purchase_conversion_rate"] = 0.0
            features["cart_conversion_rate"] = 0.0
        
        # Cart abandonment rate
        if features["cart_count"] > 0:
            features["cart_abandonment_rate"] = 1 - (features["purchase_count"] / features["cart_count"])
        else:
            features["cart_abandonment_rate"] = 0.0
        
        # Browse-to-buy ratio
        if features["purchase_count"] > 0:
            features["browse_to_buy_ratio"] = features["view_count"] / features["purchase_count"]
        else:
            features["browse_to_buy_ratio"] = float("inf") if features["view_count"] > 0 else 0.0
        
        # === DIVERSITY METRICS ===
        # Product diversity (unique products per event)
        features["product_diversity"] = features["unique_products_viewed"] / max(features["total_events"], 1)
        
        # Category diversity (unique categories per event)
        features["category_diversity"] = features["unique_categories"] / max(features["total_events"], 1)
        
        # Brand diversity
        features["brand_diversity"] = features["unique_brands"] / max(features["total_events"], 1)
        
        # Category concentration (HHI-like measure)
        if len(user_data) > 0:
            category_counts = user_data["category_id"].value_counts()
            category_proportions = category_counts / len(user_data)
            features["category_concentration"] = (category_proportions ** 2).sum()
        else:
            features["category_concentration"] = 0.0
        
        # === CATEGORY PREFERENCES ===
        # Top category (most viewed/purchased)
        if len(user_data) > 0:
            top_category = user_data["category_id"].mode()
            features["top_category_id"] = top_category[0] if len(top_category) > 0 else user_data["category_id"].iloc[0]
        else:
            features["top_category_id"] = 0
        
        # === SESSION METRICS ===
        # Average events per session
        if features["unique_sessions"] > 0:
            features["avg_events_per_session"] = features["total_events"] / features["unique_sessions"]
        else:
            features["avg_events_per_session"] = 0.0
        
        # Session duration (for sessions with multiple events)
        session_durations = []
        for session_id, session_data in user_data.groupby("user_session"):
            if len(session_data) > 1:
                session_duration = (session_data["event_time"].max() - session_data["event_time"].min()).total_seconds() / 3600
                session_durations.append(session_duration)
        
        if len(session_durations) > 0:
            features["avg_session_duration_hours"] = np.mean(session_durations)
            features["max_session_duration_hours"] = np.max(session_durations)
        else:
            features["avg_session_duration_hours"] = 0.0
            features["max_session_duration_hours"] = 0.0
        
        # === PRICE SENSITIVITY ===
        # Price sensitivity: standard deviation of prices viewed
        if len(viewed_data) > 0:
            features["price_std"] = viewed_data["price"].std()
            features["price_range"] = viewed_data["price"].max() - viewed_data["price"].min()
        else:
            features["price_std"] = 0.0
            features["price_range"] = 0.0
        
        customer_features.append(features)
    
    # Convert to DataFrame
    customer_df = pd.DataFrame(customer_features)
    
    # Convert datetime columns to numeric (days since first event)
    if "first_event_time" in customer_df.columns:
        reference_time = customer_df["first_event_time"].min()
        customer_df["days_since_first_event"] = (customer_df["first_event_time"] - reference_time).dt.days
        customer_df["days_since_last_event"] = (customer_df["last_event_time"] - reference_time).dt.days
    
    # Drop datetime columns (keep numeric versions)
    datetime_cols = ["first_event_time", "last_event_time"]
    customer_df = customer_df.drop(columns=[col for col in datetime_cols if col in customer_df.columns])
    
    logger.info(f"Aggregated to {len(customer_df)} customers with {len(customer_df.columns)} features")
    logger.info(f"Feature columns: {list(customer_df.columns)}")
    
    return customer_df


def create_rfm_features(customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create RFM (Recency, Frequency, Monetary) features.
    
    Args:
        customer_df: Customer-level DataFrame
    
    Returns:
        DataFrame with RFM features added
    """
    logger.info("Creating RFM features")
    
    df = customer_df.copy()
    
    # Recency: Days since last event (using days_since_last_event if available)
    if "days_since_last_event" in df.columns:
        df["recency"] = df["days_since_last_event"]
    else:
        df["recency"] = 0  # All events same day
    
    # Frequency: Purchase count (or total events as proxy)
    df["frequency"] = df["purchase_count"]
    df["frequency_events"] = df["total_events"]
    
    # Monetary: Total spending
    df["monetary"] = df["total_spending"]
    
    # RFM Scores (1-5 scale based on quartiles)
    for metric in ["recency", "frequency", "monetary"]:
        if metric in df.columns:
            # For recency, lower is better (more recent), so reverse
            if metric == "recency":
                df[f"{metric}_score"] = pd.qcut(df[metric], q=5, labels=False, duplicates="drop") + 1
                df[f"{metric}_score"] = 6 - df[f"{metric}_score"]  # Reverse: 5 = most recent
            else:
                df[f"{metric}_score"] = pd.qcut(df[metric], q=5, labels=False, duplicates="drop") + 1
    
    logger.info("RFM features created")
    return df


def aggregate_data(df: pd.DataFrame, include_rfm: bool = True) -> pd.DataFrame:
    """
    Complete aggregation pipeline: customer features + RFM.
    
    Args:
        df: Event-level DataFrame
        include_rfm: Whether to include RFM features
    
    Returns:
        Aggregated customer-level DataFrame
    """
    logger.info("Starting data aggregation pipeline")
    
    # Aggregate to customer level
    customer_df = aggregate_customer_features(df)
    
    # Add RFM features
    if include_rfm:
        customer_df = create_rfm_features(customer_df)
    
    logger.info("Data aggregation pipeline completed successfully")
    return customer_df


if __name__ == "__main__":
    # Test aggregation
    import pandas as pd
    from src.data.data_loader import load_and_prepare_data
    
    # Load sample data
    df = load_and_prepare_data(max_rows=10000)
    
    # Aggregate
    customer_df = aggregate_data(df)
    
    print(f"Aggregated to {len(customer_df)} customers")
    print(customer_df.head())
    print(customer_df.describe())

