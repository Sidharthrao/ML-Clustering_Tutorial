# E-commerce Customer Segmentation - Clustering ML Pipeline

An industry-ready unsupervised learning pipeline for customer segmentation using e-commerce behavioral data.

## Project Overview

This project implements a complete ML pipeline for customer segmentation using:
- **Dataset**: E-commerce behavioral data from SQLite database (1M events, 146K users)
- **Task**: Unsupervised clustering for customer segmentation
- **Algorithms**: K-means, DBSCAN, Hierarchical clustering
- **Goal**: Identify distinct customer groups based on purchasing patterns and behavior

## Project Structure

```
ECommerce behavior data/
├── config/              # Configuration parameters
│   ├── __init__.py
│   └── config.py
├── src/                 # Source code
│   ├── data/           # Data loading and aggregation
│   │   ├── data_loader.py
│   │   └── data_aggregator.py
│   ├── preprocessing/  # Feature engineering and preprocessing
│   │   ├── feature_engineering.py
│   │   └── preprocessor.py
│   ├── clustering/     # Clustering algorithms
│   │   ├── cluster_selector.py
│   │   └── cluster_trainer.py
│   ├── evaluation/     # Cluster evaluation
│   │   └── cluster_evaluator.py
│   └── utils/          # Utility functions
│       ├── logger.py
│       └── visualizations.py
├── scripts/            # Training scripts
│   └── train_clusters.py
├── api/                # Flask API endpoints (future)
├── models/             # Saved model artifacts
├── notebooks/          # Jupyter notebooks
│   └── Ecommerce_Customer_Segmentation_Pipeline.ipynb
├── reports/            # Evaluation reports and plots
├── logs/               # Application logs
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup Instructions

### 1. Virtual Environment Setup

The project uses a shared virtual environment at the repository root:

```bash
# Navigate to repository root
cd /path/to/Project-Rogue

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate      # On Windows
```

### 2. Install Dependencies

```bash
# Navigate to project directory
cd "Inttrvu/Capstone_Projects/Captsone_Project - Clustering/ECommerce behavior data"

# Install requirements
pip install -r requirements.txt
```

### 3. Database Setup

The database file is located at:
```
Inttrvu/Capstone_Projects/Database.db
```

The project automatically uses this database. Ensure the `Ecommerce_data` table exists with the following columns:
- `event_time` (TEXT/datetime)
- `event_type` (TEXT: view, cart, purchase)
- `product_id` (INTEGER)
- `category_id` (INTEGER)
- `category_code` (TEXT, nullable)
- `brand` (TEXT, nullable)
- `price` (REAL)
- `user_id` (INTEGER)
- `user_session` (TEXT)

## Usage

### Running the Complete Pipeline (Notebook)

The recommended way to run the complete pipeline with detailed explanations:

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/Ecommerce_Customer_Segmentation_Pipeline.ipynb
```

The notebook includes:
- Step-by-step execution with justifications
- Comprehensive EDA
- Feature engineering explanations
- Clustering model training and evaluation
- Visualizations and business insights

### Running the Training Script

For automated training without the notebook:

```bash
python scripts/train_clusters.py
```

This will:
1. Load data from database
2. Aggregate to customer level
3. Engineer features
4. Preprocess data
5. Determine optimal clusters
6. Train clustering model
7. Evaluate and generate reports

### Model Artifacts

After training, the following artifacts are saved:

- `models/preprocessor.pkl` - Fitted preprocessing pipeline
- `models/clusterer.pkl` - Trained clustering model
- `models/feature_names.pkl` - Feature names for consistency
- `reports/cluster_report.md` - Evaluation report
- `reports/cluster_distribution.png` - Cluster size visualization
- `reports/cluster_profiles.png` - Cluster feature profiles
- `reports/pca_clusters_2d.png` - PCA visualization
- `reports/cluster_selection_comparison.csv` - Cluster selection metrics

## Key Features

### Data Processing
- Efficient chunking for large datasets (1M+ events)
- Event-level to customer-level aggregation
- Comprehensive feature engineering (RFM, behavioral, temporal)

### Feature Engineering
- **RFM Analysis**: Recency, Frequency, Monetary scores
- **Behavioral Features**: Engagement intensity, loyalty scores
- **Temporal Features**: Peak activity hours, session patterns
- **Engagement Features**: Exploration vs focus, diversity metrics
- **Price Sensitivity**: Price preference and tolerance

### Clustering Algorithms
- **K-means**: Primary algorithm with optimal K selection
- **DBSCAN**: Density-based clustering for comparison
- **Hierarchical**: Agglomerative clustering for comparison

### Evaluation Metrics
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Cluster size distribution
- Cluster profiles and characteristics

### Preprocessing
- Outlier detection and treatment (IQR method)
- Feature scaling (RobustScaler)
- Correlation-based feature selection
- Variance-based feature selection
- Optional PCA for dimensionality reduction

## Configuration

Key configuration parameters are in `config/config.py`:

- **Clustering**: Number of clusters range, algorithm parameters
- **Preprocessing**: Scaling method, outlier thresholds
- **Feature Engineering**: RFM thresholds, temporal windows
- **Evaluation**: Metrics to calculate, visualization settings

## Output and Results

### Cluster Profiles

Each cluster is characterized by:
- Event counts (views, carts, purchases)
- Spending metrics (total, average order value)
- Engagement levels (sessions, products viewed)
- Conversion rates
- RFM scores
- Temporal patterns

### Business Insights

The pipeline identifies customer segments such as:
- **High-Value Customers**: High spending, frequent purchases
- **At-Risk Customers**: Declining engagement
- **New Customers**: Low engagement but potential
- **Price-Sensitive Customers**: Low price tolerance
- **Explorers**: High product diversity, low conversion

### Recommendations

Based on cluster analysis:
1. Personalized marketing campaigns by segment
2. Retention strategies for high-value customers
3. Re-engagement campaigns for at-risk segments
4. Pricing strategies for price-sensitive segments
5. Product recommendations based on cluster preferences

## Project Highlights

- **Industry-Ready**: Modular structure, comprehensive logging, error handling
- **Scalable**: Handles 1M+ events efficiently with chunking
- **Comprehensive**: Full pipeline from data loading to business insights
- **Well-Documented**: Detailed justifications and explanations in notebook
- **Production-Ready**: Model persistence, reproducible preprocessing

## Limitations and Future Work

- **Single-day data**: Limited temporal analysis (all events on same day)
- **No external validation**: Consider A/B testing cluster-based strategies
- **Static clustering**: Could implement online/streaming clustering for real-time updates
- **Feature expansion**: Could add product recommendations, seasonal patterns

## Troubleshooting

### Database Connection Issues
- Verify database path in `config/config.py`
- Ensure database file exists and is accessible
- Check table name matches `Ecommerce_data`

### Memory Issues
- Reduce `chunk_size` in `DATA_CONFIG`
- Use `sample_fraction` to work with subset of data
- Consider using `max_rows` parameter for testing

### Clustering Issues
- Adjust `n_clusters_range` if optimal K is at boundary
- Try different preprocessing methods (standard vs robust scaling)
- Check for sufficient variance in features (may need feature selection)

## License

This project is part of the Project-Rogue repository.

## Contact

For questions or issues, please refer to the main repository documentation.

