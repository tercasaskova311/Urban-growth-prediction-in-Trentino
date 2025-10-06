# Urban Growth Prediction in Trentino

Detect urban changes in Trentino over the last 5 years and predict which areas are likely to grow, shrink, or remain stable, based on multi-temporal Sentinel-2 imagery and auxiliary datasets.

## Project Overview

This project implements a comprehensive framework for analyzing and predicting urban growth patterns in the Trentino region of Italy. By treating each tile of the map as a **spatio-temporal signal**, where pixel values (NDVI, NDBI, urban fraction, mobility) represent the signal amplitude and time is the independent variable, we can detect trends, abrupt changes, and correlations.

## Key Features

- **Multi-source Data Integration**: Combines Sentinel-2 satellite imagery, OpenStreetMap data, and mobility datasets
- **Signal Processing Approach**: Treats urban data as spatio-temporal signals for advanced analysis
- **Trend Detection**: Identifies long-term trends in urban development
- **Change Point Detection**: Detects abrupt changes in urban patterns
- **Predictive Modeling**: Predicts future growth, shrinkage, or stability of urban areas
- **Machine Learning**: Supports multiple ML models (Random Forest, Gradient Boosting, XGBoost)

## Data Sources

### Satellite Imagery
- **Source**: Google Earth Engine (GEE) → Sentinel-2
- **Indices**: NDVI, NDBI, NDWI, Urban Index
- **Format**: GeoTIFF or cloud-optimized format

### Buildings and Roads
- **Source**: OpenStreetMap (OSM) / Trentino GIS
- **Features**: Buildings, roads, land use
- **Format**: Shapefiles, GeoPackage

### Population/Mobility
- **Sources**: 
  - Facebook Movement Range
  - Google Mobility Reports
- **Format**: CSV

## Installation

### Prerequisites

- Python 3.8 or higher
- Google Earth Engine account (for satellite data)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tercasaskova311/Urban-growth-prediction-in-Trentino.git
cd Urban-growth-prediction-in-Trentino
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Authenticate with Google Earth Engine (if using Sentinel-2 data):
```bash
earthengine authenticate
```

## Project Structure

```
Urban-growth-prediction-in-Trentino/
├── config/
│   └── config.yaml              # Configuration file
├── data/                        # Data directory (created automatically)
│   ├── raw/
│   │   ├── sentinel2/          # Sentinel-2 imagery
│   │   ├── osm/                # OpenStreetMap data
│   │   └── mobility/           # Mobility data
│   ├── processed/              # Processed features
│   └── models/                 # Trained models
├── src/
│   ├── data_acquisition/       # Data download modules
│   │   ├── sentinel2_downloader.py
│   │   ├── osm_downloader.py
│   │   └── mobility_downloader.py
│   ├── signal_processing/      # Signal processing modules
│   │   └── temporal_analysis.py
│   ├── models/                 # ML models
│   │   └── urban_growth_model.py
│   └── utils/                  # Utility functions
│       ├── config_utils.py
│       └── visualization.py
├── notebooks/                   # Jupyter notebooks
├── results/                     # Analysis results
├── figures/                     # Generated figures
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage

### Quick Start

Run the full pipeline with synthetic data:

```bash
python main.py --synthetic
```

### Running Individual Steps

1. **Data Acquisition**:
```bash
python main.py --mode acquire
```

2. **Signal Processing**:
```bash
python main.py --mode process
```

3. **Model Training**:
```bash
python main.py --mode train --synthetic
```

4. **Prediction**:
```bash
python main.py --mode predict
```

### Configuration

Edit `config/config.yaml` to customize:
- Study area bounding box
- Time period for analysis
- Sentinel-2 parameters
- Model hyperparameters
- Output directories

## Methodology

### Signal Processing Concept

Each tile of the map is treated as a **spatio-temporal signal**:
- **Signal Amplitude**: Pixel values (NDVI, NDBI, urban fraction, mobility)
- **Independent Variable**: Time
- **Analysis**: Trend detection, change point detection, correlation analysis

### Analysis Pipeline

1. **Data Acquisition**
   - Download Sentinel-2 imagery for the study area
   - Calculate spectral indices (NDVI, NDBI, NDWI)
   - Acquire OSM data (buildings, roads, land use)
   - Download mobility data

2. **Signal Processing**
   - Grid the study area into tiles
   - Extract time series for each tile
   - Detect trends using linear regression or Mann-Kendall test
   - Identify change points using CUSUM or Pettitt test
   - Calculate correlations between variables

3. **Feature Extraction**
   - NDVI trend (vegetation changes)
   - NDBI trend (built-up area changes)
   - Building density
   - Road density
   - Mobility index

4. **Model Training**
   - Train classification model (Random Forest, Gradient Boosting, XGBoost)
   - Predict three classes: **grow**, **shrink**, **stable**
   - Evaluate using cross-validation

5. **Prediction and Visualization**
   - Generate predictions for future urban growth
   - Create spatial maps of predictions
   - Visualize trends and change points

## Examples

### Example 1: Trend Detection

```python
from src.signal_processing import SpatioTemporalProcessor
import numpy as np

# Create processor
processor = SpatioTemporalProcessor()

# Example time series (NDBI values over time)
time_series = np.array([0.1, 0.12, 0.15, 0.18, 0.22, 0.25])
timestamps = np.array([0, 1, 2, 3, 4, 5])

# Detect trend
trend = processor.detect_trend(time_series, timestamps)
print(f"Slope: {trend['slope']:.4f}, R²: {trend['r_squared']:.4f}")
```

### Example 2: Training a Model

```python
from src.models import UrbanGrowthPredictor, create_synthetic_dataset

# Create synthetic dataset
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
```

## Results

The model predicts urban growth patterns and classifies each tile as:
- **Grow**: Areas expected to experience urban expansion
- **Stable**: Areas expected to remain unchanged
- **Shrink**: Areas expected to experience urban decline or reforestation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this project in your research, please cite:

```
Urban Growth Prediction in Trentino
https://github.com/tercasaskova311/Urban-growth-prediction-in-Trentino
```

## Acknowledgments

- Sentinel-2 data provided by ESA Copernicus program via Google Earth Engine
- OpenStreetMap contributors for geographic data
- Google Mobility Reports and Facebook Data for Good for mobility data

## Contact

For questions or collaborations, please open an issue on GitHub.