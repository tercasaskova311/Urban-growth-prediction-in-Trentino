"""
Urban growth prediction model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import pickle


class UrbanGrowthPredictor:
    """
    Class for predicting urban growth patterns.
    Predicts whether areas will grow, shrink, or remain stable.
    """
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Initialize the urban growth predictor.
        
        Args:
            model_type: Type of model ('gradient_boosting', 'random_forest', 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_names = ['shrink', 'stable', 'grow']
    
    def prepare_features(self, data: pd.DataFrame, 
                        feature_columns: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for training/prediction.
        
        Args:
            data: DataFrame with features
            feature_columns: List of feature column names
            
        Returns:
            Tuple of (feature array, feature names)
        """
        # Select feature columns
        X = data[feature_columns].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        return X, feature_columns
    
    def create_labels(self, initial_values: np.ndarray, 
                     final_values: np.ndarray,
                     threshold: float = 0.1) -> np.ndarray:
        """
        Create labels based on change between initial and final values.
        
        Args:
            initial_values: Initial urban index values
            final_values: Final urban index values
            threshold: Threshold for defining significant change
            
        Returns:
            Array of labels (0=shrink, 1=stable, 2=grow)
        """
        change = (final_values - initial_values) / (initial_values + 1e-6)
        
        labels = np.zeros(len(change), dtype=int)
        labels[change < -threshold] = 0  # Shrink
        labels[np.abs(change) <= threshold] = 1  # Stable
        labels[change > threshold] = 2  # Grow
        
        return labels
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             feature_names: Optional[List[str]] = None,
             test_size: float = 0.2, **model_params) -> Dict:
        """
        Train the prediction model.
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: Names of features
            test_size: Proportion of test set
            **model_params: Additional model parameters
            
        Returns:
            Dictionary with training results
        """
        self.feature_names = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 10),
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 6),
                learning_rate=model_params.get('learning_rate', 0.1),
                random_state=42
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 6),
                learning_rate=model_params.get('learning_rate', 0.1),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if feature_names:
                feature_importance = dict(zip(feature_names, importance))
                print("\nFeature Importance:")
                for feat, imp in sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {feat}: {imp:.4f}")
        
        return {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data['class_names']
        
        print(f"Model loaded from {filepath}")
    
    def evaluate_spatial_predictions(self, predictions: np.ndarray,
                                    coordinates: np.ndarray) -> Dict:
        """
        Evaluate spatial distribution of predictions.
        
        Args:
            predictions: Array of predictions
            coordinates: Array of (x, y) coordinates
            
        Returns:
            Dictionary with spatial statistics
        """
        stats = {}
        
        for class_id, class_name in enumerate(self.class_names):
            mask = predictions == class_id
            stats[class_name] = {
                'count': int(mask.sum()),
                'percentage': float(mask.sum() / len(predictions) * 100)
            }
            
            if mask.sum() > 0:
                coords = coordinates[mask]
                stats[class_name]['mean_x'] = float(coords[:, 0].mean())
                stats[class_name]['mean_y'] = float(coords[:, 1].mean())
                stats[class_name]['std_x'] = float(coords[:, 0].std())
                stats[class_name]['std_y'] = float(coords[:, 1].std())
        
        return stats


def create_synthetic_dataset(n_samples: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create a synthetic dataset for testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (features DataFrame, labels)
    """
    np.random.seed(42)
    
    # Generate features
    data = {
        'ndvi_trend': np.random.normal(0, 0.1, n_samples),
        'ndbi_trend': np.random.normal(0, 0.1, n_samples),
        'building_density': np.random.uniform(0, 100, n_samples),
        'road_density': np.random.uniform(0, 50, n_samples),
        'mobility_index': np.random.normal(0, 20, n_samples),
        'ndvi_mean': np.random.uniform(0.2, 0.8, n_samples),
        'ndbi_mean': np.random.uniform(-0.5, 0.5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create labels based on features
    # Areas with positive NDBI trend and high building density tend to grow
    # Areas with positive NDVI trend and low building density tend to shrink (reforestation)
    labels = np.ones(n_samples, dtype=int)  # Default: stable
    
    growth_mask = (df['ndbi_trend'] > 0.05) & (df['building_density'] > 50)
    labels[growth_mask] = 2  # Grow
    
    shrink_mask = (df['ndvi_trend'] > 0.05) & (df['building_density'] < 20)
    labels[shrink_mask] = 0  # Shrink
    
    return df, labels
