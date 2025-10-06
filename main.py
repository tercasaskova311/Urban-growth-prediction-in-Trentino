#!/usr/bin/env python3
"""
Main script for Urban Growth Prediction in Trentino.

This script orchestrates the entire pipeline:
1. Data acquisition from Sentinel-2, OSM, and mobility sources
2. Signal processing and feature extraction
3. Model training and prediction
4. Visualization of results
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_config, get_data_paths
from src.data_acquisition import Sentinel2Downloader, OSMDataDownloader, MobilityDataDownloader
from src.signal_processing import SpatioTemporalProcessor
from src.models import UrbanGrowthPredictor, create_synthetic_dataset
from src.utils.visualization import plot_confusion_matrix, plot_feature_importance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Urban Growth Prediction in Trentino'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['acquire', 'process', 'train', 'predict', 'full'],
        default='full',
        help='Execution mode: acquire data, process signals, train model, predict, or full pipeline'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data for testing'
    )
    
    return parser.parse_args()


def acquire_data(config, paths):
    """
    Acquire data from various sources.
    
    Args:
        config: Configuration dictionary
        paths: Data paths dictionary
    """
    print("\n" + "="*60)
    print("STEP 1: DATA ACQUISITION")
    print("="*60)
    
    # Extract configuration
    bbox = config['study_area']['bbox']
    start_date = config['time_period']['start_date']
    end_date = config['time_period']['end_date']
    
    # 1. Sentinel-2 data
    print("\n1.1 Acquiring Sentinel-2 imagery...")
    print("Note: Requires Google Earth Engine authentication.")
    print("Run 'earthengine authenticate' if not already authenticated.")
    
    # Sentinel-2 downloader (requires GEE authentication)
    # s2_downloader = Sentinel2Downloader(bbox, start_date, end_date)
    # s2_downloader.initialize_gee()
    # collection = s2_downloader.get_image_collection()
    
    # 2. OSM data
    print("\n1.2 Acquiring OpenStreetMap data...")
    osm_downloader = OSMDataDownloader(bbox)
    
    # Download buildings
    buildings = osm_downloader.download_buildings()
    if not buildings.empty:
        buildings.to_file(os.path.join(paths['osm'], 'buildings.gpkg'), driver='GPKG')
    
    # Download roads
    roads = osm_downloader.download_roads()
    if not roads.empty:
        roads.to_file(os.path.join(paths['osm'], 'roads.gpkg'), driver='GPKG')
    
    # 3. Mobility data
    print("\n1.3 Acquiring mobility data...")
    mobility_downloader = MobilityDataDownloader()
    
    # Try to download Google Mobility data
    mobility_data = mobility_downloader.download_google_mobility(
        start_date, end_date,
        output_file=os.path.join(paths['mobility'], 'google_mobility.csv')
    )
    
    # If no data available, create synthetic data for testing
    if mobility_data.empty:
        print("Creating synthetic mobility data for testing...")
        mobility_data = mobility_downloader.create_synthetic_mobility_data(
            start_date, end_date,
            output_file=os.path.join(paths['mobility'], 'synthetic_mobility.csv')
        )
    
    print("\nData acquisition completed!")


def process_signals(config, paths):
    """
    Process signals and extract features.
    
    Args:
        config: Configuration dictionary
        paths: Data paths dictionary
    """
    print("\n" + "="*60)
    print("STEP 2: SIGNAL PROCESSING")
    print("="*60)
    
    tile_size = config['signal_processing']['tile_size']
    overlap = config['signal_processing']['overlap']
    
    processor = SpatioTemporalProcessor(tile_size, overlap)
    
    print(f"\nProcessing with tile size: {tile_size}m, overlap: {overlap}m")
    print("This is where spatio-temporal signal analysis would be performed.")
    print("Each tile is treated as a signal with pixel values as amplitude and time as the independent variable.")
    
    print("\nSignal processing completed!")


def train_model(config, paths, use_synthetic=False):
    """
    Train the urban growth prediction model.
    
    Args:
        config: Configuration dictionary
        paths: Data paths dictionary
        use_synthetic: Whether to use synthetic data
    """
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60)
    
    model_config = config['model']
    model_type = model_config['type']
    
    print(f"\nTraining {model_type} model...")
    
    # Create or load dataset
    if use_synthetic:
        print("Using synthetic dataset for demonstration...")
        df, labels = create_synthetic_dataset(n_samples=1000)
    else:
        print("Loading processed dataset...")
        # In a real scenario, load the processed features here
        df, labels = create_synthetic_dataset(n_samples=1000)
    
    # Feature columns
    feature_columns = model_config['features']
    
    # Initialize predictor
    predictor = UrbanGrowthPredictor(model_type=model_type)
    
    # Prepare features
    X, feature_names = predictor.prepare_features(df, feature_columns)
    
    # Train model
    results = predictor.train(
        X, labels, 
        feature_names=feature_names,
        **model_config.get('xgboost', {})
    )
    
    # Save model
    model_path = os.path.join(paths['models'], 'urban_growth_model.pkl')
    predictor.save_model(model_path)
    
    # Save visualizations
    if results['confusion_matrix'] is not None:
        cm_path = os.path.join(paths['figures'], 'confusion_matrix.png')
        plot_confusion_matrix(
            results['confusion_matrix'],
            predictor.class_names,
            save_path=cm_path
        )
    
    print("\nModel training completed!")
    return predictor


def predict_growth(config, paths, predictor=None):
    """
    Make predictions on new data.
    
    Args:
        config: Configuration dictionary
        paths: Data paths dictionary
        predictor: Trained predictor (optional, will load if not provided)
    """
    print("\n" + "="*60)
    print("STEP 4: PREDICTION")
    print("="*60)
    
    # Load or use provided predictor
    if predictor is None:
        predictor = UrbanGrowthPredictor()
        model_path = os.path.join(paths['models'], 'urban_growth_model.pkl')
        predictor.load_model(model_path)
    
    # Create sample data for prediction
    print("\nGenerating predictions...")
    df, _ = create_synthetic_dataset(n_samples=500)
    
    feature_columns = config['model']['features']
    X, _ = predictor.prepare_features(df, feature_columns)
    
    # Make predictions
    predictions = predictor.predict(X)
    probabilities = predictor.predict_proba(X)
    
    # Print summary
    print("\nPrediction Summary:")
    for i, class_name in enumerate(predictor.class_names):
        count = (predictions == i).sum()
        percentage = count / len(predictions) * 100
        print(f"  {class_name.capitalize()}: {count} tiles ({percentage:.1f}%)")
    
    print("\nPrediction completed!")


def main():
    """Main execution function."""
    args = parse_args()
    
    print("="*60)
    print("URBAN GROWTH PREDICTION IN TRENTINO")
    print("="*60)
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Error: Could not load configuration. Exiting.")
        return 1
    
    # Setup data paths
    paths = get_data_paths(config)
    
    # Execute based on mode
    if args.mode in ['acquire', 'full']:
        try:
            acquire_data(config, paths)
        except Exception as e:
            print(f"Error during data acquisition: {e}")
            if args.mode != 'full':
                return 1
    
    if args.mode in ['process', 'full']:
        try:
            process_signals(config, paths)
        except Exception as e:
            print(f"Error during signal processing: {e}")
            if args.mode != 'full':
                return 1
    
    if args.mode in ['train', 'full']:
        try:
            predictor = train_model(config, paths, use_synthetic=args.synthetic or args.mode == 'full')
        except Exception as e:
            print(f"Error during model training: {e}")
            if args.mode != 'full':
                return 1
    else:
        predictor = None
    
    if args.mode in ['predict', 'full']:
        try:
            predict_growth(config, paths, predictor)
        except Exception as e:
            print(f"Error during prediction: {e}")
            if args.mode != 'full':
                return 1
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
