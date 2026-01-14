"""
Data loading module.
Handles database connection and data extraction from SQLite database for e-commerce data.
"""
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import DATABASE_PATH, DB_TABLE_NAME, DATA_CONFIG
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")


def load_data_from_db(
    db_path: Optional[Path] = None,
    table_name: Optional[str] = None,
    chunk_size: int = 100000,
    max_rows: Optional[int] = None,
    sample_fraction: Optional[float] = None
) -> pd.DataFrame:
    """
    Load data from SQLite database with chunking for large datasets.
    
    Args:
        db_path: Path to database file (defaults to config DATABASE_PATH)
        table_name: Name of table to load (defaults to config DB_TABLE_NAME)
        chunk_size: Number of rows to load per chunk
        max_rows: Maximum number of rows to load (None for all)
        sample_fraction: Fraction of data to sample (None for all)
    
    Returns:
        DataFrame containing the loaded data
    """
    if db_path is None:
        db_path = DATABASE_PATH
    if table_name is None:
        table_name = DB_TABLE_NAME
    
    logger.info(f"Loading data from {db_path} table {table_name}")
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path))
        
        # Get total row count
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        logger.info(f"Total rows in table: {total_rows:,}")
        
        # Determine how many rows to load
        rows_to_load = total_rows
        if max_rows:
            rows_to_load = min(total_rows, max_rows)
        elif sample_fraction:
            rows_to_load = int(total_rows * sample_fraction)
        
        logger.info(f"Loading {rows_to_load:,} rows")
        
        # Load data in chunks
        chunks = []
        offset = 0
        
        while offset < rows_to_load:
            current_chunk_size = min(chunk_size, rows_to_load - offset)
            
            # Use ORDER BY for consistent sampling if needed
            if sample_fraction and sample_fraction < 1.0:
                # Use random sampling via SQL
                query = f"""
                    SELECT * FROM {table_name} 
                    ORDER BY RANDOM() 
                    LIMIT {current_chunk_size}
                """
            else:
                query = f"SELECT * FROM {table_name} LIMIT {current_chunk_size} OFFSET {offset}"
            
            chunk = pd.read_sql_query(query, conn)
            chunks.append(chunk)
            
            offset += current_chunk_size
            logger.debug(f"Loaded {offset:,}/{rows_to_load:,} rows")
        
        conn.close()
        
        # Concatenate all chunks
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Successfully loaded {len(df):,} rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column types according to e-commerce data schema.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with converted types
    """
    logger.info("Converting column types")
    df_converted = df.copy()
    
    # Convert event_time to datetime
    if "event_time" in df_converted.columns:
        df_converted["event_time"] = pd.to_datetime(df_converted["event_time"], errors="coerce")
        logger.debug("Converted event_time to datetime")
    
    # Convert numeric columns
    numeric_columns = ["product_id", "category_id", "user_id", "price"]
    for col in numeric_columns:
        if col in df_converted.columns:
            df_converted[col] = pd.to_numeric(df_converted[col], errors="coerce")
            logger.debug(f"Converted {col} to numeric")
    
    # Convert string columns
    string_columns = ["event_type", "category_code", "brand", "user_session"]
    for col in string_columns:
        if col in df_converted.columns:
            df_converted[col] = df_converted[col].astype("string")
            logger.debug(f"Converted {col} to string")
    
    return df_converted


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data validation checks for e-commerce data.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Validated DataFrame
    
    Raises:
        ValueError: If critical validation checks fail
    """
    logger.info("Validating data")
    
    # Check required columns
    required_columns = [
        "event_time", "event_type", "product_id", "category_id",
        "user_id", "user_session"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Log basic statistics
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Check event types
    if "event_type" in df.columns:
        event_counts = df["event_type"].value_counts()
        logger.info(f"Event type distribution:\n{event_counts}")
    
    # Check for invalid prices
    if "price" in df.columns:
        negative_prices = (df["price"] < 0).sum()
        if negative_prices > 0:
            logger.warning(f"Found {negative_prices} records with negative prices")
    
    # Check for invalid timestamps
    if "event_time" in df.columns:
        invalid_times = df["event_time"].isna().sum()
        if invalid_times > 0:
            logger.warning(f"Found {invalid_times} records with invalid timestamps")
    
    # Check for duplicate events
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate records")
    
    return df


def load_and_prepare_data(
    db_path: Optional[Path] = None,
    table_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    max_rows: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    convert_types: bool = True,
    validate: bool = True
) -> pd.DataFrame:
    """
    Complete data loading pipeline: load, convert types, and validate.
    
    Args:
        db_path: Path to database file
        table_name: Name of table to load
        chunk_size: Number of rows to load per chunk
        max_rows: Maximum number of rows to load
        sample_fraction: Fraction of data to sample
        convert_types: Whether to convert column types
        validate: Whether to validate data
    
    Returns:
        Prepared DataFrame
    """
    logger.info("Starting data loading pipeline")
    
    # Use config defaults if not provided
    if chunk_size is None:
        chunk_size = DATA_CONFIG.get("chunk_size", 100000)
    if max_rows is None:
        max_rows = DATA_CONFIG.get("max_rows", None)
    if sample_fraction is None:
        sample_fraction = DATA_CONFIG.get("sample_fraction", None)
    
    # Load data
    df = load_data_from_db(db_path, table_name, chunk_size, max_rows, sample_fraction)
    
    # Convert types
    if convert_types:
        df = convert_column_types(df)
    
    # Validate
    if validate:
        df = validate_data(df)
    
    logger.info("Data loading pipeline completed successfully")
    return df


if __name__ == "__main__":
    # Test data loading
    df = load_and_prepare_data(max_rows=1000)
    print(f"Loaded {len(df)} rows")
    print(df.head())
    print(df.info())

