"""
Preprocessing pipeline module for clustering.
Handles outlier detection, scaling, feature selection, and dimensionality reduction.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import joblib
from typing import Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import PREPROCESSING_CONFIG, FEATURE_CONFIG
from src.utils.logger import setup_logger

logger = setup_logger("preprocessor")


class ClusteringPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible preprocessor for customer clustering.
    Handles missing values, outlier treatment, scaling, and feature selection.
    """
    
    def __init__(
        self,
        scaling_method: str = "robust",
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.01,
        outlier_iqr_multiplier: float = 1.5,
        outlier_cap_percentiles: List[float] = [5, 95],
        use_pca: bool = False,
        pca_variance_explained: float = 0.95,
        missing_value_strategy: str = "median"
    ):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: 'standard' or 'robust' scaler
            correlation_threshold: Threshold for removing highly correlated features
            variance_threshold: Threshold for removing low variance features
            outlier_iqr_multiplier: IQR multiplier for outlier detection
            outlier_cap_percentiles: Percentiles for capping outliers
            use_pca: Whether to apply PCA for dimensionality reduction
            pca_variance_explained: Variance to retain in PCA
            missing_value_strategy: Strategy for imputation ('mean', 'median', 'mode', 'drop')
        """
        self.scaling_method = scaling_method
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.outlier_cap_percentiles = outlier_cap_percentiles
        self.use_pca = use_pca
        self.pca_variance_explained = pca_variance_explained
        self.missing_value_strategy = missing_value_strategy
        
        # Initialize transformers
        if scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.imputer = SimpleImputer(strategy=missing_value_strategy if missing_value_strategy != "drop" else "median")
        self.variance_selector = VarianceThreshold(threshold=variance_threshold)
        self.pca = PCA(n_components=pca_variance_explained) if use_pca else None
        
        # Store state
        self.feature_names_ = None
        self.correlated_features_ = []
        self.is_fitted_ = False
        
    def _detect_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using IQR method.
        
        Args:
            X: Input DataFrame
        
        Returns:
            DataFrame with outliers capped
        """
        logger.info("Detecting and handling outliers")
        X_processed = X.copy()
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = X_processed[col].quantile(0.25)
            Q3 = X_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - self.outlier_iqr_multiplier * IQR
                upper_bound = Q3 + self.outlier_iqr_multiplier * IQR
                
                # Cap outliers
                X_processed[col] = X_processed[col].clip(lower=lower_bound, upper=upper_bound)
                
                # Also cap by percentiles if specified
                if self.outlier_cap_percentiles:
                    lower_percentile = X_processed[col].quantile(self.outlier_cap_percentiles[0] / 100)
                    upper_percentile = X_processed[col].quantile(self.outlier_cap_percentiles[1] / 100)
                    X_processed[col] = X_processed[col].clip(lower=lower_percentile, upper=upper_percentile)
        
        logger.info("Outlier handling completed")
        return X_processed
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Args:
            X: Input DataFrame
        
        Returns:
            DataFrame with correlated features removed
        """
        logger.info(f"Removing features with correlation > {self.correlation_threshold}")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return X
        
        # Calculate correlation matrix
        corr_matrix = X[numeric_cols].corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.correlation_threshold)]
        
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
            self.correlated_features_ = to_drop
            X = X.drop(columns=to_drop)
        else:
            logger.info("No highly correlated features found")
        
        return X
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ClusteringPreprocessor':
        """
        Fit the preprocessor on data.
        
        Args:
            X: Feature DataFrame
            y: Not used (for sklearn compatibility)
        
        Returns:
            Self
        """
        logger.info("Fitting preprocessor")
        
        # Select only numeric columns
        X_numeric = X.select_dtypes(include=[np.number]).copy()
        
        if X_numeric.empty:
            raise ValueError("No numeric columns found for preprocessing")
        
        logger.info(f"Processing {len(X_numeric.columns)} numeric features")
        
        # Handle missing values
        if X_numeric.isnull().any().any():
            logger.info("Imputing missing values")
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X_numeric),
                columns=X_numeric.columns,
                index=X_numeric.index
            )
        else:
            X_imputed = X_numeric.copy()
        
        # Detect and handle outliers
        X_processed = self._detect_outliers(X_imputed)
        
        # Remove highly correlated features
        X_processed = self._remove_correlated_features(X_processed)
        
        # Remove low variance features
        logger.info(f"Removing features with variance < {self.variance_threshold}")
        X_processed_variance = self.variance_selector.fit_transform(X_processed)
        selected_features = X_processed.columns[self.variance_selector.get_support()]
        X_processed = pd.DataFrame(
            X_processed_variance,
            columns=selected_features,
            index=X_processed.index
        )
        logger.info(f"Selected {len(selected_features)} features after variance filtering")
        
        # Fit scaler
        logger.info(f"Fitting {self.scaling_method} scaler")
        self.scaler.fit(X_processed)
        
        # Fit PCA if enabled
        if self.use_pca and self.pca is not None:
            logger.info(f"Fitting PCA to explain {self.pca_variance_explained*100}% variance")
            X_scaled = self.scaler.transform(X_processed)
            self.pca.fit(X_scaled)
            n_components = self.pca.n_components_
            logger.info(f"PCA will reduce to {n_components} components")
        
        # Store feature names
        if self.use_pca and self.pca is not None:
            self.feature_names_ = [f"PC{i+1}" for i in range(self.pca.n_components_)]
        else:
            self.feature_names_ = list(selected_features)
        
        self.is_fitted_ = True
        logger.info(f"Preprocessor fitted. Output features: {len(self.feature_names_)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.info("Transforming data")
        
        # Select only numeric columns
        X_numeric = X.select_dtypes(include=[np.number]).copy()
        
        # Remove correlated features (if they exist)
        if self.correlated_features_:
            X_numeric = X_numeric.drop(columns=[col for col in self.correlated_features_ if col in X_numeric.columns])
        
        # Handle missing values
        if X_numeric.isnull().any().any():
            X_imputed = pd.DataFrame(
                self.imputer.transform(X_numeric),
                columns=X_numeric.columns,
                index=X_numeric.index
            )
        else:
            X_imputed = X_numeric.copy()
        
        # Handle outliers
        X_processed = self._detect_outliers(X_imputed)
        
        # Remove low variance features
        X_processed = pd.DataFrame(
            self.variance_selector.transform(X_processed),
            columns=X_processed.columns[self.variance_selector.get_support()],
            index=X_processed.index
        )
        
        # Scale
        X_scaled = self.scaler.transform(X_processed)
        X_scaled = pd.DataFrame(
            X_scaled,
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        # Apply PCA if enabled
        if self.use_pca and self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
            X_scaled = pd.DataFrame(
                X_scaled,
                columns=self.feature_names_,
                index=X_processed.index
            )
        
        logger.info(f"Transformed to {X_scaled.shape}")
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def save(self, filepath: Path) -> None:
        """Save preprocessor to disk."""
        logger.info(f"Saving preprocessor to {filepath}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Preprocessor saved")
    
    @staticmethod
    def load(filepath: Path) -> 'ClusteringPreprocessor':
        """Load preprocessor from disk."""
        logger.info(f"Loading preprocessor from {filepath}")
        return joblib.load(filepath)


if __name__ == "__main__":
    # Test preprocessor
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    
    # Add some missing values and outliers
    X.iloc[0:10, 0] = np.nan
    X.iloc[100:110, 1] = 1000  # Outliers
    
    # Test preprocessor
    preprocessor = ClusteringPreprocessor(
        scaling_method="robust",
        use_pca=False
    )
    
    X_transformed = preprocessor.fit_transform(X)
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Feature names: {preprocessor.feature_names_}")

