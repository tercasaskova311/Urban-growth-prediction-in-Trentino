#!/usr/bin/env python3
"""
Quick example demonstrating the urban growth prediction framework.
This example uses synthetic data and demonstrates the core functionality.
"""

import sys
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from src.signal_processing import SpatioTemporalProcessor
from src.models import UrbanGrowthPredictor, create_synthetic_dataset


def example_signal_processing():
    """Demonstrate signal processing capabilities."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Signal Processing for Temporal Analysis")
    print("="*70)
    
    # Create a processor
    processor = SpatioTemporalProcessor(tile_size=1000, overlap=100)
    
    # Example: NDBI time series showing urban growth (5 years of data)
    time_series = np.array([0.10, 0.12, 0.15, 0.18, 0.22, 0.25])
    timestamps = np.array([0, 1, 2, 3, 4, 5])  # Years
    
    print("\nTime series data (NDBI values over 5 years):")
    for i, (t, val) in enumerate(zip(timestamps, time_series)):
        print(f"  Year {t}: {val:.3f}")
    
    # Detect trend
    print("\n1. Trend Detection (Linear Regression):")
    trend = processor.detect_trend(time_series, timestamps, method='linear_regression')
    print(f"   Slope: {trend['slope']:.4f} (positive = urbanization)")
    print(f"   R²: {trend['r_squared']:.4f} (goodness of fit)")
    print(f"   P-value: {trend['p_value']:.4f}")
    
    if trend['p_value'] < 0.05:
        print("   ✓ Statistically significant urbanization trend detected!")
    
    # Detect change points
    print("\n2. Change Point Detection (CUSUM):")
    change_points = processor.detect_change_points(time_series, method='cusum')
    if change_points:
        print(f"   Detected change points at indices: {change_points}")
    else:
        print("   No abrupt changes detected (smooth urbanization)")
    
    # Extract features
    print("\n3. Feature Extraction:")
    features = processor.extract_features(time_series, timestamps)
    print(f"   Mean NDBI: {features['mean']:.4f}")
    print(f"   Std Dev: {features['std']:.4f}")
    print(f"   Range: {features['range']:.4f}")
    print(f"   Trend slope: {features['trend_slope']:.4f}")
    print(f"   Coefficient of variation: {features['coefficient_of_variation']:.4f}")


def example_model_training():
    """Demonstrate model training and prediction."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Urban Growth Prediction Model")
    print("="*70)
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    df, labels = create_synthetic_dataset(n_samples=1000)
    
    print(f"   Dataset size: {len(df)} tiles")
    print(f"   Features: {', '.join(df.columns.tolist())}")
    
    # Show label distribution
    class_names = ['Shrink', 'Stable', 'Grow']
    print("\n   Label distribution:")
    for i, name in enumerate(class_names):
        count = (labels == i).sum()
        pct = count / len(labels) * 100
        print(f"     {name}: {count} tiles ({pct:.1f}%)")
    
    # Train model
    print("\n2. Training Gradient Boosting model...")
    predictor = UrbanGrowthPredictor(model_type='gradient_boosting')
    
    feature_columns = [
        'ndvi_trend',        # Vegetation trend
        'ndbi_trend',        # Built-up area trend
        'building_density',  # Current building density
        'road_density',      # Road network density
        'mobility_index'     # Mobility patterns
    ]
    
    X, feature_names = predictor.prepare_features(df, feature_columns)
    
    # Train with smaller model for demo
    results = predictor.train(
        X, labels,
        feature_names=feature_names,
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1
    )
    
    print(f"\n   Model accuracy: {results['accuracy']:.2%}")
    
    # Make predictions on new data
    print("\n3. Making predictions on new data...")
    test_df, _ = create_synthetic_dataset(n_samples=200)
    X_test, _ = predictor.prepare_features(test_df, feature_columns)
    predictions = predictor.predict(X_test)
    
    print("\n   Prediction distribution:")
    for i, name in enumerate(predictor.class_names):
        count = (predictions == i).sum()
        pct = count / len(predictions) * 100
        print(f"     {name.capitalize()}: {count} tiles ({pct:.1f}%)")


def example_correlation_analysis():
    """Demonstrate correlation analysis between variables."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Correlation Analysis")
    print("="*70)
    
    processor = SpatioTemporalProcessor()
    
    # Example: NDBI increasing while NDVI decreasing (urban expansion)
    ndbi_series = np.array([0.1, 0.15, 0.18, 0.22, 0.25, 0.28])
    ndvi_series = np.array([0.6, 0.55, 0.50, 0.45, 0.42, 0.38])
    
    print("\nAnalyzing relationship between NDBI and NDVI:")
    print("  NDBI (Built-up): ", [f"{v:.2f}" for v in ndbi_series])
    print("  NDVI (Vegetation):", [f"{v:.2f}" for v in ndvi_series])
    
    correlation = processor.calculate_correlation(ndbi_series, ndvi_series)
    
    print(f"\nCorrelation results:")
    print(f"  Pearson correlation: {correlation['pearson']:.4f}")
    print(f"  P-value: {correlation['pearson_p_value']:.4f}")
    
    if correlation['pearson'] < -0.7:
        print("  ✓ Strong negative correlation: Urban growth replacing vegetation")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("URBAN GROWTH PREDICTION IN TRENTINO")
    print("Quick Examples Using Synthetic Data")
    print("="*70)
    
    try:
        example_signal_processing()
        example_model_training()
        example_correlation_analysis()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run 'python main.py --synthetic' for full pipeline")
        print("  2. Configure GEE credentials for real Sentinel-2 data")
        print("  3. Adjust config/config.yaml for your study area")
        print("  4. See notebooks/quick_start_example.ipynb for more details")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
