# Usage Guide

This guide explains how to use the Urban Growth Prediction framework for analyzing and predicting urban changes in Trentino.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/tercasaskova311/Urban-growth-prediction-in-Trentino.git
cd Urban-growth-prediction-in-Trentino

# Install dependencies
pip install -r requirements.txt

# Optional: Install geospatial packages for full functionality
pip install earthengine-api osmnx geopandas rasterio
```

### 2. Run Examples

```bash
# Run interactive examples with synthetic data
python examples.py

# Run the full pipeline with synthetic data
python main.py --synthetic
```

### 3. Explore the Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/quick_start_example.ipynb
```

## Configuration

Edit `config/config.yaml` to customize the analysis:

```yaml
# Study area (Trentino bounding box)
study_area:
  bbox: [10.4, 45.7, 12.0, 47.1]  # [min_lon, min_lat, max_lon, max_lat]

# Time period
time_period:
  start_date: "2018-01-01"
  end_date: "2023-12-31"

# Model parameters
model:
  type: "gradient_boosting"  # or "random_forest", "xgboost"
  features:
    - "ndvi_trend"
    - "ndbi_trend"
    - "building_density"
    - "road_density"
    - "mobility_index"
```

## Usage Modes

### Mode 1: Data Acquisition

Download satellite imagery, OSM data, and mobility data:

```bash
python main.py --mode acquire
```

**Note**: Requires Google Earth Engine authentication:
```bash
earthengine authenticate
```

### Mode 2: Signal Processing

Process time series data and extract features:

```bash
python main.py --mode process
```

### Mode 3: Model Training

Train the urban growth prediction model:

```bash
python main.py --mode train
```

### Mode 4: Prediction

Generate predictions for urban growth:

```bash
python main.py --mode predict
```

### Mode 5: Full Pipeline

Run all steps in sequence:

```bash
python main.py --mode full
```

## Python API

### Signal Processing

```python
from src.signal_processing import SpatioTemporalProcessor
import numpy as np

# Create processor
processor = SpatioTemporalProcessor()

# Detect trends in NDBI time series
time_series = np.array([0.1, 0.12, 0.15, 0.18, 0.22, 0.25])
timestamps = np.array([0, 1, 2, 3, 4, 5])

trend = processor.detect_trend(time_series, timestamps)
print(f"Urbanization slope: {trend['slope']:.4f}")
print(f"R²: {trend['r_squared']:.4f}")

# Detect change points
change_points = processor.detect_change_points(time_series)
print(f"Change points: {change_points}")

# Extract features
features = processor.extract_features(time_series, timestamps)
```

### Urban Growth Prediction

```python
from src.models import UrbanGrowthPredictor, create_synthetic_dataset

# Create or load dataset
df, labels = create_synthetic_dataset(n_samples=1000)

# Initialize predictor
predictor = UrbanGrowthPredictor(model_type='gradient_boosting')

# Define features
features = ['ndvi_trend', 'ndbi_trend', 'building_density', 
           'road_density', 'mobility_index']

# Prepare and train
X, feature_names = predictor.prepare_features(df, features)
results = predictor.train(X, labels, feature_names=feature_names)

# Save model
predictor.save_model('models/my_model.pkl')

# Make predictions
predictions = predictor.predict(X_new)
```

### Data Acquisition

#### Sentinel-2 Data

```python
from src.data_acquisition import Sentinel2Downloader

# Initialize downloader
bbox = [10.4, 45.7, 12.0, 47.1]  # Trentino
downloader = Sentinel2Downloader(bbox, "2018-01-01", "2023-12-31")

# Authenticate with GEE
downloader.initialize_gee()

# Get image collection
collection = downloader.get_image_collection(cloud_cover=20)

# Export to GeoTIFF
# (requires additional setup)
```

#### OpenStreetMap Data

```python
from src.data_acquisition import OSMDataDownloader

# Initialize downloader
bbox = [10.4, 45.7, 12.0, 47.1]
downloader = OSMDataDownloader(bbox)

# Download buildings
buildings = downloader.download_buildings()
buildings.to_file('data/buildings.gpkg', driver='GPKG')

# Download roads
roads = downloader.download_roads()
roads.to_file('data/roads.gpkg', driver='GPKG')

# Calculate building density
density = downloader.calculate_building_density(grid_size=1000)
```

#### Mobility Data

```python
from src.data_acquisition import MobilityDataDownloader

# Initialize downloader
downloader = MobilityDataDownloader(region="Trentino")

# Download Google Mobility data (if available)
mobility = downloader.download_google_mobility(
    start_date="2020-01-01",
    end_date="2023-12-31",
    output_file='data/mobility.csv'
)

# Or create synthetic data for testing
synthetic = downloader.create_synthetic_mobility_data(
    start_date="2018-01-01",
    end_date="2023-12-31"
)
```

## Understanding the Signal Processing Approach

This project treats each tile of the map as a **spatio-temporal signal**:

- **Signal Amplitude**: Pixel values (NDVI, NDBI, urban fraction, mobility)
- **Independent Variable**: Time
- **Analysis Techniques**: 
  - Trend detection (linear regression, Mann-Kendall test)
  - Change point detection (CUSUM, Pettitt test)
  - Correlation analysis

### Example: Detecting Urban Growth

```python
# NDBI increasing over time indicates urbanization
ndbi_series = [0.10, 0.12, 0.15, 0.18, 0.22, 0.25]

# Positive slope = urbanization
# Negative slope = de-urbanization (rare)
# No slope = stable
```

### Example: Detecting Change Points

```python
# Abrupt change in NDBI might indicate:
# - New development project
# - Infrastructure construction
# - Land use change

ndbi_series = [0.10, 0.11, 0.12, 0.11, 0.25, 0.27, 0.28]
#                                        ↑ Change point
```

## Model Output

The model predicts one of three classes for each tile:

1. **Grow** (2): Area expected to experience urban expansion
2. **Stable** (1): Area expected to remain unchanged
3. **Shrink** (0): Area expected to experience urban decline or reforestation

### Interpretation

```python
predictions = predictor.predict(X)

# Count predictions
grow_count = (predictions == 2).sum()
stable_count = (predictions == 1).sum()
shrink_count = (predictions == 0).sum()

print(f"Growth areas: {grow_count}")
print(f"Stable areas: {stable_count}")
print(f"Shrink areas: {shrink_count}")
```

## Troubleshooting

### Google Earth Engine Authentication

If you get authentication errors:

```bash
earthengine authenticate
```

Follow the prompts to authenticate with your Google account.

### Missing Dependencies

If you encounter import errors:

```bash
# Install all optional dependencies
pip install earthengine-api osmnx geopandas rasterio xgboost
```

### Memory Issues

For large study areas, reduce the number of tiles or increase tile size in `config.yaml`:

```yaml
signal_processing:
  tile_size: 2000  # Increase from 1000
```

## Advanced Usage

### Custom Feature Engineering

```python
# Add custom features
df['urban_change_rate'] = (df['ndbi_trend'] - df['ndvi_trend']) / 2
df['infrastructure_score'] = df['building_density'] + df['road_density']

# Use in model
custom_features = features + ['urban_change_rate', 'infrastructure_score']
```

### Ensemble Models

```python
# Train multiple models
rf_predictor = UrbanGrowthPredictor(model_type='random_forest')
gb_predictor = UrbanGrowthPredictor(model_type='gradient_boosting')
xgb_predictor = UrbanGrowthPredictor(model_type='xgboost')

# Train each
rf_predictor.train(X, y)
gb_predictor.train(X, y)
xgb_predictor.train(X, y)

# Ensemble predictions
rf_pred = rf_predictor.predict(X_test)
gb_pred = gb_predictor.predict(X_test)
xgb_pred = xgb_predictor.predict(X_test)

# Majority voting
from scipy import stats
ensemble_pred = stats.mode([rf_pred, gb_pred, xgb_pred], axis=0)[0]
```

## Next Steps

1. **Real Data**: Configure GEE and download actual Sentinel-2 data
2. **Custom Study Area**: Update bounding box in config.yaml
3. **Feature Tuning**: Experiment with different feature combinations
4. **Model Tuning**: Optimize hyperparameters
5. **Validation**: Compare predictions with ground truth data

## Support

For issues or questions:
- Open an issue on GitHub
- Check the notebooks for detailed examples
- Review the API documentation in source files
