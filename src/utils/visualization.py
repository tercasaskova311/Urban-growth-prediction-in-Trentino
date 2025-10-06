"""
Visualization utilities for urban growth analysis.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None  # type: ignore


def plot_time_series(time_series: np.ndarray, timestamps: np.ndarray,
                    title: str = "Time Series", ylabel: str = "Value",
                    save_path: Optional[str] = None):
    """
    Plot time series data.
    
    Args:
        time_series: Array of values
        timestamps: Array of timestamps
        title: Plot title
        ylabel: Y-axis label
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, time_series, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_trend_analysis(time_series: np.ndarray, timestamps: np.ndarray,
                       trend_stats: Dict, save_path: Optional[str] = None):
    """
    Plot time series with trend line.
    
    Args:
        time_series: Array of values
        timestamps: Array of timestamps
        trend_stats: Dictionary with trend statistics (slope, intercept)
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    # Plot data
    plt.plot(timestamps, time_series, marker='o', linestyle='-', 
            linewidth=2, label='Data', alpha=0.7)
    
    # Plot trend line
    trend_line = trend_stats['slope'] * timestamps + trend_stats['intercept']
    plt.plot(timestamps, trend_line, 'r--', linewidth=2, 
            label=f"Trend (slope={trend_stats['slope']:.4f})")
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f"Trend Analysis (R²={trend_stats['r_squared']:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_feature_importance(importances: Dict[str, float], top_n: int = 10,
                          save_path: Optional[str] = None):
    """
    Plot feature importance.
    
    Args:
        importances: Dictionary of feature importances
        top_n: Number of top features to show
        save_path: Optional path to save figure
    """
    # Sort by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, values = zip(*sorted_features)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(features)), values)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_spatial_predictions(gdf: "gpd.GeoDataFrame", 
                            prediction_column: str = 'prediction',
                            class_names: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
    """
    Plot spatial distribution of predictions.
    
    Args:
        gdf: GeoDataFrame with predictions
        prediction_column: Name of prediction column
        class_names: Names of prediction classes
        save_path: Optional path to save figure
    """
    if not HAS_GEOPANDAS:
        print("GeoPandas is required for spatial plotting")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    if class_names:
        cmap = plt.cm.get_cmap('RdYlGn', len(class_names))
    else:
        cmap = 'viridis'
    
    gdf.plot(column=prediction_column, ax=ax, legend=True, cmap=cmap,
            edgecolor='black', linewidth=0.1)
    
    ax.set_title('Urban Growth Predictions', fontsize=16)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    if class_names:
        # Add custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cmap(i/len(class_names)), 
                                label=name)
                          for i, name in enumerate(class_names)]
        ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_index_comparison(df: pd.DataFrame, indices: List[str],
                         save_path: Optional[str] = None):
    """
    Plot comparison of multiple indices over time.
    
    Args:
        df: DataFrame with time series data
        indices: List of index column names
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(len(indices), 1, figsize=(12, 4*len(indices)))
    
    if len(indices) == 1:
        axes = [axes]
    
    for ax, index in zip(axes, indices):
        if index in df.columns:
            df[index].plot(ax=ax, linewidth=2)
            ax.set_ylabel(index)
            ax.set_title(f'{index} Over Time')
            ax.grid(True, alpha=0.3)
    
    plt.xlabel('Time')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
