"""
Signal processing module for spatio-temporal analysis.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression


class SpatioTemporalProcessor:
    """
    Class for processing spatio-temporal signals in urban data.
    Each tile is treated as a signal with pixel values as amplitude and time as independent variable.
    """
    
    def __init__(self, tile_size: int = 1000, overlap: int = 100):
        """
        Initialize the spatio-temporal processor.
        
        Args:
            tile_size: Size of each tile in meters
            overlap: Overlap between tiles in meters
        """
        self.tile_size = tile_size
        self.overlap = overlap
    
    def detect_trend(self, time_series: np.ndarray, timestamps: np.ndarray,
                    method: str = 'linear_regression') -> Dict:
        """
        Detect trend in time series data.
        
        Args:
            time_series: Array of values over time
            timestamps: Array of timestamps (or time indices)
            method: Method for trend detection ('linear_regression', 'mann_kendall')
            
        Returns:
            Dictionary with trend statistics
        """
        if method == 'linear_regression':
            return self._linear_trend(time_series, timestamps)
        elif method == 'mann_kendall':
            return self._mann_kendall_trend(time_series)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _linear_trend(self, time_series: np.ndarray, 
                     timestamps: np.ndarray) -> Dict:
        """
        Calculate linear trend using least squares regression.
        
        Args:
            time_series: Array of values over time
            timestamps: Array of timestamps
            
        Returns:
            Dictionary with slope, intercept, and R²
        """
        # Remove NaN values
        mask = ~np.isnan(time_series)
        if mask.sum() < 2:
            return {'slope': 0, 'intercept': 0, 'r_squared': 0, 'p_value': 1}
        
        x = timestamps[mask].reshape(-1, 1)
        y = time_series[mask]
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(x, y)
        
        # Calculate R² and p-value
        y_pred = model.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate p-value for slope
        n = len(x)
        if n > 2:
            se = np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((x - x.mean()) ** 2))
            t_stat = model.coef_[0] / se if se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            p_value = 1
        
        return {
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'r_squared': float(r_squared),
            'p_value': float(p_value)
        }
    
    def _mann_kendall_trend(self, time_series: np.ndarray) -> Dict:
        """
        Mann-Kendall trend test (non-parametric).
        
        Args:
            time_series: Array of values over time
            
        Returns:
            Dictionary with trend statistics
        """
        # Remove NaN values
        data = time_series[~np.isnan(time_series)]
        n = len(data)
        
        if n < 2:
            return {'trend': 'no trend', 'tau': 0, 'p_value': 1}
        
        # Calculate Mann-Kendall statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(data[j] - data[i])
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate z-score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Determine trend
        if p_value < 0.05:
            trend = 'increasing' if s > 0 else 'decreasing'
        else:
            trend = 'no trend'
        
        # Calculate Kendall's tau
        tau = s / (n * (n - 1) / 2)
        
        return {
            'trend': trend,
            'tau': float(tau),
            'p_value': float(p_value),
            'z_score': float(z)
        }
    
    def detect_change_points(self, time_series: np.ndarray, 
                           method: str = 'cusum') -> List[int]:
        """
        Detect abrupt changes in time series.
        
        Args:
            time_series: Array of values over time
            method: Method for change detection ('cusum', 'pettitt')
            
        Returns:
            List of change point indices
        """
        if method == 'cusum':
            return self._cusum_change_detection(time_series)
        elif method == 'pettitt':
            return self._pettitt_test(time_series)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _cusum_change_detection(self, time_series: np.ndarray,
                               threshold: float = 5) -> List[int]:
        """
        CUSUM (Cumulative Sum) change detection.
        
        Args:
            time_series: Array of values over time
            threshold: Threshold for change detection
            
        Returns:
            List of change point indices
        """
        # Remove NaN values
        data = time_series[~np.isnan(time_series)]
        
        if len(data) < 2:
            return []
        
        # Calculate mean and std
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return []
        
        # Calculate CUSUM
        cusum_pos = np.zeros(len(data))
        cusum_neg = np.zeros(len(data))
        
        for i in range(1, len(data)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + (data[i] - mean) / std - 0.5)
            cusum_neg[i] = min(0, cusum_neg[i-1] + (data[i] - mean) / std + 0.5)
        
        # Find change points
        change_points = []
        for i in range(len(data)):
            if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                change_points.append(i)
        
        return change_points
    
    def _pettitt_test(self, time_series: np.ndarray) -> List[int]:
        """
        Pettitt test for change point detection.
        
        Args:
            time_series: Array of values over time
            
        Returns:
            List with single change point index (if significant)
        """
        # Remove NaN values
        data = time_series[~np.isnan(time_series)]
        n = len(data)
        
        if n < 2:
            return []
        
        # Calculate U statistic for each potential change point
        U = np.zeros(n)
        for t in range(n):
            for i in range(t + 1):
                for j in range(t + 1, n):
                    U[t] += np.sign(data[j] - data[i])
        
        # Find maximum |U|
        K = np.argmax(np.abs(U))
        
        # Calculate p-value (approximate)
        p_value = 2 * np.exp(-6 * U[K]**2 / (n**3 + n**2))
        
        # Return change point if significant
        if p_value < 0.05:
            return [K]
        else:
            return []
    
    def calculate_correlation(self, series1: np.ndarray, 
                            series2: np.ndarray) -> Dict:
        """
        Calculate correlation between two time series.
        
        Args:
            series1: First time series
            series2: Second time series
            
        Returns:
            Dictionary with correlation statistics
        """
        # Remove NaN values
        mask = ~(np.isnan(series1) | np.isnan(series2))
        s1 = series1[mask]
        s2 = series2[mask]
        
        if len(s1) < 2:
            return {'pearson': 0, 'spearman': 0, 'p_value': 1}
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(s1, s2)
        
        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(s1, s2)
        
        return {
            'pearson': float(pearson_r),
            'pearson_p_value': float(pearson_p),
            'spearman': float(spearman_r),
            'spearman_p_value': float(spearman_p)
        }
    
    def smooth_time_series(self, time_series: np.ndarray, 
                          window_size: int = 3) -> np.ndarray:
        """
        Smooth time series using moving average.
        
        Args:
            time_series: Array of values over time
            window_size: Window size for moving average
            
        Returns:
            Smoothed time series
        """
        return np.convolve(time_series, np.ones(window_size)/window_size, mode='same')
    
    def extract_features(self, time_series: np.ndarray,
                        timestamps: np.ndarray) -> Dict:
        """
        Extract multiple features from time series.
        
        Args:
            time_series: Array of values over time
            timestamps: Array of timestamps
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        # Statistical features
        features['mean'] = float(np.nanmean(time_series))
        features['std'] = float(np.nanstd(time_series))
        features['min'] = float(np.nanmin(time_series))
        features['max'] = float(np.nanmax(time_series))
        features['range'] = features['max'] - features['min']
        
        # Trend features
        trend = self.detect_trend(time_series, timestamps)
        features['trend_slope'] = trend['slope']
        features['trend_r_squared'] = trend['r_squared']
        
        # Change detection
        change_points = self.detect_change_points(time_series)
        features['n_change_points'] = len(change_points)
        features['has_change'] = len(change_points) > 0
        
        # Variability features
        if len(time_series) > 1:
            features['coefficient_of_variation'] = features['std'] / features['mean'] if features['mean'] != 0 else 0
        else:
            features['coefficient_of_variation'] = 0
        
        return features
