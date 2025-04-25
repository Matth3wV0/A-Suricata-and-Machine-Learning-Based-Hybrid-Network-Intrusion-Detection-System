import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, Tuple


class AnomalyDetector:
    """
    Provides ML and statistical anomaly detection capabilities with support for enriched features
    """
    
    def __init__(self, model_dir='./model'):
        """
        Initialize the anomaly detector with trained models and baseline
        
        Args:
            model_dir: Directory containing trained models and baseline
        """
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None
        self.baseline = {}
        
        # Load models and baseline
        self._load_models()
    
    def _load_models(self):
        """Load trained models and baseline from disk"""
        try:
            # Load Decision Tree model
            with open(os.path.join(self.model_dir, 'dt_model.pkl'), 'rb') as f:
                import pickle
                self.models['dt_model'] = pickle.load(f)
            
            # Load Random Forest model
            with open(os.path.join(self.model_dir, 'rf_model.pkl'), 'rb') as f:
                import pickle
                self.models['rf_model'] = pickle.load(f)
            
            # Try to load XGBoost model if available
            try:
                with open(os.path.join(self.model_dir, 'xgb_model.pkl'), 'rb') as f:
                    import pickle
                    self.models['xgb_model'] = pickle.load(f)
            except FileNotFoundError:
                pass  # XGBoost model is optional
            
            # Load scaler
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                import pickle
                self.scaler = pickle.load(f)
            
            # Load baseline statistics
            with open(os.path.join(self.model_dir, 'baseline.json'), 'r') as f:
                self.baseline = json.load(f)
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def detect_ml_anomaly(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies using machine learning models
        
        Args:
            features: DataFrame containing flow features
            
        Returns:
            Dictionary with detection results
        """
        if not self.models or 'dt_model' not in self.models or 'rf_model' not in self.models:
            return {'is_anomalous': False, 'score': 0}
        
        try:
            # Separate base features from enriched features
            base_features = self._extract_base_features(features)
            
            # Ensure we have all required base features
            required_features = [col for col in self.models['dt_model'].feature_names_in_ 
                              if col in base_features.columns]
            
            # Check if we have enough features to make a prediction
            if len(required_features) < len(self.models['dt_model'].feature_names_in_) * 0.7:
                print(f"Warning: Not enough features present. Have {len(required_features)}, need at least {len(self.models['dt_model'].feature_names_in_) * 0.7}")
                return {'is_anomalous': False, 'score': 0, 'dt_prediction': 0, 'rf_prediction': 0}
            
            # Create a DataFrame with only the required features in the expected order
            features_subset = base_features[required_features].copy()
            
            # For any missing features, fill with zeros
            for feature in self.models['dt_model'].feature_names_in_:
                if feature not in features_subset.columns:
                    features_subset[feature] = 0
            
            # Ensure columns are in the same order as during training
            features_subset = features_subset[self.models['dt_model'].feature_names_in_]
            
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(features_subset)
                features_scaled_df = pd.DataFrame(features_scaled, 
                                              columns=self.models['dt_model'].feature_names_in_)
            else:
                features_scaled_df = features_subset
            
            # Get predictions from models
            dt_pred = self.models['dt_model'].predict(features_scaled_df)[0]
            rf_pred = self.models['rf_model'].predict(features_scaled_df)[0]
            
            # Get prediction probabilities
            dt_proba = self.models['dt_model'].predict_proba(features_scaled_df)[0]
            rf_proba = self.models['rf_model'].predict_proba(features_scaled_df)[0]
            
            # Calculate anomaly score from base models
            dt_score = dt_proba[1] if len(dt_proba) > 1 else 0
            rf_score = rf_proba[1] if len(rf_proba) > 1 else 0
            
            # Initialize base result
            result = {
                'dt_prediction': int(dt_pred),
                'rf_prediction': int(rf_pred),
                'dt_confidence': float(dt_score),
                'rf_confidence': float(rf_score)
            }
            
            # Add XGBoost if available
            if 'xgb_model' in self.models:
                xgb_pred = self.models['xgb_model'].predict(features_scaled_df)[0]
                xgb_proba = self.models['xgb_model'].predict_proba(features_scaled_df)[0]
                xgb_score = xgb_proba[1] if len(xgb_proba) > 1 else 0
                
                result['xgb_prediction'] = int(xgb_pred)
                result['xgb_confidence'] = float(xgb_score)
                
                # Calculate combined score with XGBoost
                combined_score = (dt_score * 0.2 + rf_score * 0.3 + xgb_score * 0.5)
                
                # Voting logic
                votes = int(dt_pred) + int(rf_pred) + int(xgb_pred)
                is_anomalous = votes >= 2  # At least 2 out of 3 models agree
            else:
                # Combined score without XGBoost
                combined_score = (dt_score * 0.4 + rf_score * 0.6)
                
                # Voting logic
                votes = int(dt_pred) + int(rf_pred)
                is_anomalous = votes >= 1  # At least 1 out of 2 models predict anomaly
            
            # Skip app-layer detection for now since we're troubleshooting
            # Just use the base ML detection results
            
            result['score'] = float(combined_score)
            result['is_anomalous'] = bool(is_anomalous)
            
            return result
            
        except Exception as e:
            print(f"Error in ML anomaly detection: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {'is_anomalous': False, 'score': 0, 'dt_prediction': 0, 'rf_prediction': 0}
    
    def _extract_base_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Extract base features used by ML models"""
        # These are the core features that our models were trained on
        base_feature_columns = [
            'dest_port', 'duration', 'total_fwd_packets', 'total_bwd_packets',
            'total_fwd_bytes', 'total_bwd_bytes', 'flow_bytes_per_sec',
            'flow_packets_per_sec', 'down_up_ratio'
        ]
        
        # Get available base features
        available_columns = [col for col in base_feature_columns if col in features.columns]
        
        # Return only the base features
        return features[available_columns]
    
    def _analyze_app_layer_features(self, features: pd.DataFrame) -> float:
        """Simplified function that just returns 0 for now"""
        # We'll simplify this function to avoid the error for now
        # You can re-enable it later once the base system works
        return 0.0
    
    def detect_statistical_anomaly(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies using statistical methods
        
        Args:
            features: DataFrame containing flow features
            
        Returns:
            Dictionary with detection results and details
        """
        if not self.baseline:
            return {'is_anomalous': False, 'score': 0, 'details': []}
        
        try:
            # Extract base features for statistical analysis
            base_features = self._extract_base_features(features)
            
            # Scale features if scaler is available - using a safer approach
            if self.scaler:
                # Scale only columns that are in the scaler
                scaler_columns = base_features.columns.intersection(
                    self.models['dt_model'].feature_names_in_)
                
                if not scaler_columns.empty:
                    # Store original dtypes before scaling
                    original_dtypes = {col: base_features[col].dtype for col in scaler_columns}
                    
                    # Do the scaling
                    scaled_values = self.scaler.transform(base_features[scaler_columns])
                    features_scaled_df = pd.DataFrame(scaled_values, columns=scaler_columns)
                    
                    # Create a new copy to avoid SettingWithCopyWarning
                    safe_base_features = base_features.copy()
                    
                    # Update values while respecting original dtypes
                    for col in scaler_columns:
                        if col in features_scaled_df.columns:
                            # For integer columns, we need to round
                            if np.issubdtype(original_dtypes[col], np.integer):
                                safe_base_features[col] = features_scaled_df[col].round().astype(original_dtypes[col])
                            else:
                                safe_base_features[col] = features_scaled_df[col].astype(original_dtypes[col])
                    
                    # Use the updated dataframe
                    base_features = safe_base_features
            # Initialize anomaly score and details
            anomaly_score = 0
            anomaly_details = []
            feature_scores = {}
            
            # Check each feature against the baseline
            for feature in base_features.columns:
                if feature not in self.baseline:
                    continue
                
                try:
                    # Get feature value safely
                    if len(base_features) > 0:
                        value = float(base_features[feature].iloc[0])
                    else:
                        continue
                    
                    # Get baseline stats
                    stats = self.baseline[feature]
                    
                    # Calculate Z-score if std > 0
                    if stats['std'] > 0:
                        z_score = (value - stats['mean']) / stats['std']
                    else:
                        z_score = 0
                    
                    # Store feature z-score
                    feature_scores[feature] = abs(z_score)
                    
                    # Check for outlier based on IQR
                    iqr = stats['iqr']
                    if iqr > 0:
                        lower_bound = stats['q1'] - 1.5 * iqr
                        upper_bound = stats['q3'] + 1.5 * iqr
                        is_outlier = value < lower_bound or value > upper_bound
                    else:
                        is_outlier = False
                    
                    # Use a higher z-score threshold (4.0 instead of 3.0) to reduce false positives
                    z_threshold = 4.0
                    
                    # Add to anomaly score with more conservative threshold
                    if abs(z_score) > z_threshold or is_outlier:
                        # Weight different features differently
                        feature_weight = 1.0
                        if 'flow_bytes_per_sec' in feature:
                            feature_weight = 0.8  # Reduce weight for bytes/sec which can vary a lot
                        elif 'dest_port' in feature:
                            feature_weight = 1.2  # Increase weight for destination port
                        
                        anomaly_score += (abs(z_score) if abs(z_score) > 0 else 1.0) * feature_weight
                    
                    # Add to details if anomalous (but use original threshold for reporting)
                    if abs(z_score) > 3 or is_outlier:
                        anomaly_details.append({
                            'feature': feature,
                            'value': float(value),
                            'z_score': float(z_score),
                            'is_outlier': is_outlier,
                            'baseline_mean': stats['mean'],
                            'baseline_std': stats['std']
                        })
                except Exception as e:
                    print(f"Error processing feature {feature}: {e}")
                    continue
            
            # Sort details by z-score (most anomalous first)
            anomaly_details.sort(key=lambda x: abs(x['z_score']), reverse=True)
            
            # Normalize score
            if len(base_features.columns) > 0:
                anomaly_score /= len(base_features.columns)
            
            # Require at least two features to be anomalous with a more conservative threshold
            min_anomalous_features = 2
            
            # Count features with high z-scores
            high_z_features = sum(1 for score in feature_scores.values() if score > 4.0)
            
            # Determine if anomalous with a higher threshold to reduce false positives
            is_anomalous = anomaly_score > 0.7 or (high_z_features >= min_anomalous_features)
            
            # Skip app-layer detection for now
            
            return {
                'score': float(anomaly_score),
                'is_anomalous': bool(is_anomalous),
                'details': anomaly_details
            }
            
        except Exception as e:
            print(f"Error in statistical anomaly detection: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {'is_anomalous': False, 'score': 0, 'details': []}
    
    def detect_anomalies(self, features: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        """
        Perform both ML and statistical anomaly detection
        
        Args:
            features: DataFrame containing flow features
            
        Returns:
            Tuple of (ml_result, stat_result, combined_score)
        """
        # Detect using ML models
        ml_result = self.detect_ml_anomaly(features)
        
        # Detect using statistical methods
        stat_result = self.detect_statistical_anomaly(features)
        
        # Combine scores from both methods
        combined_score = max(ml_result.get('score', 0), stat_result.get('score', 0))
        
        return ml_result, stat_result, combined_score