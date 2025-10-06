"""
Urban Growth Prediction in Trentino

A comprehensive framework for detecting and predicting urban changes in the Trentino region
using multi-temporal Sentinel-2 imagery, OpenStreetMap data, and mobility datasets.
"""

__version__ = "0.1.0"
__author__ = "Urban Growth Prediction Team"

from .data_acquisition import Sentinel2Downloader, OSMDataDownloader, MobilityDataDownloader
from .signal_processing import SpatioTemporalProcessor
from .models import UrbanGrowthPredictor, create_synthetic_dataset
from .utils import load_config, save_config, get_data_paths

__all__ = [
    'Sentinel2Downloader',
    'OSMDataDownloader',
    'MobilityDataDownloader',
    'SpatioTemporalProcessor',
    'UrbanGrowthPredictor',
    'create_synthetic_dataset',
    'load_config',
    'save_config',
    'get_data_paths'
]
