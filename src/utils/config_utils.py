"""
Utility functions for configuration and file handling.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")


def ensure_directory(directory: str):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def get_data_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get data paths from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with data paths
    """
    base_dir = config.get('output', {}).get('data_dir', 'data')
    
    paths = {
        'raw': os.path.join(base_dir, 'raw'),
        'processed': os.path.join(base_dir, 'processed'),
        'sentinel2': os.path.join(base_dir, 'raw', 'sentinel2'),
        'osm': os.path.join(base_dir, 'raw', 'osm'),
        'mobility': os.path.join(base_dir, 'raw', 'mobility'),
        'results': config.get('output', {}).get('results_dir', 'results'),
        'figures': config.get('output', {}).get('figures_dir', 'figures'),
        'models': os.path.join(base_dir, 'models')
    }
    
    # Ensure all directories exist
    for path in paths.values():
        ensure_directory(path)
    
    return paths
