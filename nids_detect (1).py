#!/usr/bin/env python3
"""
NIDS Detection Module

This script analyzes Suricata eve.json logs to detect both known and zero-day attacks
using the trained MTH-IDS model.

Usage:
    python nids_detect.py --mode analyze --model MODEL_DIR --evejson EVE_JSON_PATH
    python nids_detect.py --mode monitor --model MODEL_DIR --evejson EVE_JSON_PATH

Arguments:
    --mode          Operation mode: 'analyze' (single file) or 'monitor' (continuous)
    --model         Path to the trained model directory
    --evejson       Path to the Suricata eve.json file
    --threshold     (Optional) Detection threshold (0.0-1.0), default 0.8
    --verbose       (Optional) Enable verbose logging

Example:
    python nids_detect.py --mode analyze --model models/ --evejson eve.json
    python nids_detect.py --mode monitor --model models/ --evejson eve.json --threshold 0.7
"""

import os
import sys
import time
import json
import pickle
import datetime
import logging
import signal
import argparse
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('nids-detect')


class NIDS_Detector:
    """
    NIDS Detector class for analyzing Suricata eve.json logs
    """
    
    def __init__(self, model_dir, eve_json_path, detection_threshold=0.8, monitoring=False):
        """
        Initialize the detector
        
        Args:
            model_dir: Path to the trained model directory
            eve_json_path: Path to the Suricata eve.json file
            detection_threshold: Threshold for detection confidence
            monitoring: Whether in monitoring mode or single analysis
        """
        self.model_dir = model_dir
        self.eve_json_path = eve_json_path
        self.detection_threshold = detection_threshold
        self.is_monitoring = monitoring
        self.last_position = 0
        
        # Load models and configuration
        self.load_model()
        
        # Initialize statistics
        self.stats = {
            'processed': 0,
            'alerts': 0,
            'start_time': datetime.datetime.now()
        }
    
    def load_model(self):
        """
        Load the trained model and other components
        """
        logger.info("Loading model and configuration...")
        
        # Load model
        model_path = os.path.join(self.model_dir, 'mth_ids_model.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
        else:
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load selected features
        features_path = os.path.join(self.model_dir, 'selected_features.json')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.selected_features = json.load(f)
            logger.info(f"Loaded {len(self.selected_features)} selected features")
        else:
            logger.warning("Selected features file not found")
            self.selected_features = []
        
        # Load baseline profile if available
        baseline_path = os.path.join(self.model_dir, 'baseline.json')
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                self.baseline = json.load(f)
            logger.info("Baseline profile loaded successfully")
        else:
            logger.warning("Baseline profile not found")
            self.baseline = None
    
    def process_eve_json(self):
        """
        Process the entire eve.json file
        
        Returns:
            DataFrame with parsed records
        """
        logger.info(f"Processing eve.json file: {self.eve_json_path}")
        
        # Read the file line by line
        records = []
        with open(self.eve_json_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse JSON
                    event = json.loads(line)
                    
                    # Skip alert records (these are intrusions already detected by Suricata)
                    if event.get('event_type') == 'alert':
                        continue
                    
                    # Add to records list
                    records.append(event)
                    
                    # Print progress for large files
                    if line_num % 10000 == 0:
                        logger.info(f"Processed {line_num} lines...")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing line {line_num}: Invalid JSON")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
        
        self.stats['processed'] += len(records)
        logger.info(f"Extracted {len(records)} valid records")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        return df
    
    def process_new_entries(self):
        """
        Process new entries added to the eve.json file since last check
        Used for real-time monitoring
        """
        if not self.eve_json_path:
            logger.error("No eve.json file specified for monitoring")
            return
        
        try:
            # Open file and move to last position
            with open(self.eve_json_path, 'r') as f:
                f.seek(self.last_position)
                
                # Read and process new lines
                new_records = []
                line_count = 0
                
                for line in f:
                    line_count += 1
                    try:
                        # Parse JSON
                        event = json.loads(line)
                        
                        # Skip alert records (these are intrusions already detected by Suricata)
                        if event.get('event_type') == 'alert':
                            continue
                        
                        # Add to records list
                        new_records.append(event)
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing JSON: Invalid format")
                    except Exception as e:
                        logger.warning(f"Error processing event: {e}")
                
                # Update last position
                self.last_position = f.tell()
            
            # Process the new records if any
            if new_records:
                logger.info(f"Processing {len(new_records)} new records...")
                
                # Create DataFrame from new records
                df = pd.DataFrame(new_records)
                self.stats['processed'] += len(new_records)
                
                # Extract features and analyze
                features = self.extract_features(df)
                self.analyze_records(features, df)
            
            elif line_count > 0:
                logger.debug(f"Read {line_count} lines, but no valid records found")
                
        except FileNotFoundError:
            logger.error(f"File not found: {self.eve_json_path}")
        except PermissionError:
            logger.error(f"Permission denied: {self.eve_json_path}")
        except Exception as e:
            logger.error(f"Error processing new entries: {e}")
    
    def extract_features(self, df):
        """
        Extract features from eve.json records to match the model's expected features
        
        Args:
            df: DataFrame with parsed eve.json records
            
        Returns:
            DataFrame with extracted features
        """
        logger.info("Extracting features from eve.json records...")
        
        # Initialize an empty DataFrame for the features
        features = pd.DataFrame(index=df.index)
        
        # Define mapping between eve.json fields and CICIDS2017 features
        feature_mapping = {
            'Flow ID': 'flow_id',
            'Source IP': 'src_ip',
            'Source Port': 'src_port',
            'Destination IP': 'dest_ip',
            'Destination Port': 'dest_port',
            'Protocol': 'proto',
            'Flow Duration': ('flow', 'age'),
            'Total Fwd Packets': ('flow', 'pkts_toserver'),
            'Total Backward Packets': ('flow', 'pkts_toclient'),
            'Total Length of Fwd Packets': ('flow', 'bytes_toserver'),
            'Total Length of Bwd Packets': ('flow', 'bytes_toclient')
        }
        
        # Extract fields from the mapping
        for cicids_field, eve_field in feature_mapping.items():
            if isinstance(eve_field, tuple):
                # Handle nested fields (e.g., flow.age)
                parent_field, child_field = eve_field
                if parent_field in df.columns:
                    features[cicids_field] = df[parent_field].apply(
                        lambda x: x.get(child_field, 0) if isinstance(x, dict) else 0
                    )
                else:
                    features[cicids_field] = 0
            elif eve_field in df.columns:
                # Handle direct fields
                features[eve_field] = df[eve_field]
            else:
                features[cicids_field] = 0
        
        # Extract time-based features
        if 'timestamp' in df.columns:
            features['timestamp'] = pd.to_datetime(df['timestamp'])
            features['Hour'] = features['timestamp'].dt.hour
            features['Minute'] = features['timestamp'].dt.minute
            features['Second'] = features['timestamp'].dt.second
        
        # Calculate derived metrics
        if 'Flow Duration' in features.columns and features['Flow Duration'].sum() > 0:
            # Flow rates
            features['Flow Bytes/s'] = (features['Total Length of Fwd Packets'] + 
                                       features['Total Length of Bwd Packets']) / features['Flow Duration'].replace(0, 1)
            features['Flow Packets/s'] = (features['Total Fwd Packets'] + 
                                        features['Total Backward Packets']) / features['Flow Duration'].replace(0, 1)
        else:
            features['Flow Bytes/s'] = 0
            features['Flow Packets/s'] = 0
        
        # Extract TCP flags if available
        if 'tcp' in df.columns:
            features['FIN Flag Count'] = df['tcp'].apply(lambda x: 1 if isinstance(x, dict) and x.get('fin', False) else 0)
            features['SYN Flag Count'] = df['tcp'].apply(lambda x: 1 if isinstance(x, dict) and x.get('syn', False) else 0)
            features['RST Flag Count'] = df['tcp'].apply(lambda x: 1 if isinstance(x, dict) and x.get('rst', False) else 0)
            features['PSH Flag Count'] = df['tcp'].apply(lambda x: 1 if isinstance(x, dict) and x.get('psh', False) else 0)
            features['ACK Flag Count'] = df['tcp'].apply(lambda x: 1 if isinstance(x, dict) and x.get('ack', False) else 0)
            features['URG Flag Count'] = df['tcp'].apply(lambda x: 1 if isinstance(x, dict) and x.get('urg', False) else 0)
        else:
            features['FIN Flag Count'] = 0
            features['SYN Flag Count'] = 0
            features['RST Flag Count'] = 0
            features['PSH Flag Count'] = 0
            features['ACK Flag Count'] = 0
            features['URG Flag Count'] = 0
        
        # Add packet length statistics (estimated from bytes and packet counts)
        if ('Total Length of Fwd Packets' in features.columns and 
            'Total Fwd Packets' in features.columns and
            'Total Length of Bwd Packets' in features.columns and
            'Total Backward Packets' in features.columns):
            
            # Forward packet length statistics
            fwd_pkt_count = features['Total Fwd Packets'].replace(0, 1)
            features['Fwd Packet Length Mean'] = features['Total Length of Fwd Packets'] / fwd_pkt_count
            features['Fwd Packet Length Min'] = features['Fwd Packet Length Mean'] * 0.5  # Estimation
            features['Fwd Packet Length Max'] = features['Fwd Packet Length Mean'] * 1.5  # Estimation
            features['Fwd Packet Length Std'] = features['Fwd Packet Length Mean'] * 0.25  # Estimation
            
            # Backward packet length statistics
            bwd_pkt_count = features['Total Backward Packets'].replace(0, 1)
            features['Bwd Packet Length Mean'] = features['Total Length of Bwd Packets'] / bwd_pkt_count
            features['Bwd Packet Length Min'] = features['Bwd Packet Length Mean'] * 0.5  # Estimation
            features['Bwd Packet Length Max'] = features['Bwd Packet Length Mean'] * 1.5  # Estimation
            features['Bwd Packet Length Std'] = features['Bwd Packet Length Mean'] * 0.25  # Estimation
            
            # Overall packet statistics
            total_packets = features['Total Fwd Packets'] + features['Total Backward Packets']
            total_bytes = features['Total Length of Fwd Packets'] + features['Total Length of Bwd Packets']
            features['Packet Length Mean'] = total_bytes / total_packets.replace(0, 1)
            features['Packet Length Min'] = features['Packet Length Mean'] * 0.5  # Estimation
            features['Packet Length Max'] = features['Packet Length Mean'] * 1.5  # Estimation
            features['Packet Length Std'] = features['Packet Length Mean'] * 0.25  # Estimation
            features['Packet Length Variance'] = features['Packet Length Std'] ** 2
            
            # IAT (Inter-Arrival Time) estimations
            if 'Flow Duration' in features.columns and features['Flow Duration'].sum() > 0:
                # Estimate average IAT
                avg_iat = features['Flow Duration'] / total_packets.replace(0, 1)
                features['Flow IAT Mean'] = avg_iat
                features['Flow IAT Min'] = avg_iat * 0.5  # Estimation
                features['Flow IAT Max'] = avg_iat * 1.5  # Estimation
                features['Flow IAT Std'] = avg_iat * 0.25  # Estimation
            
            # Header length estimations (typical TCP/IP header is ~40 bytes)
            header_size = 40  # Typical TCP/IP header size
            features['Fwd Header Length'] = header_size * features['Total Fwd Packets']
            features['Bwd Header Length'] = header_size * features['Total Backward Packets']
        
        # Fill missing values with zeros
        features = features.fillna(0)
        
        # Ensure all required features are present
        for feature in self.selected_features:
            if feature not in features.columns:
                features[feature] = 0
        
        logger.info(f"Extracted {len(features.columns)} features from {len(features)} records")
        
        return features
    
    def predict_with_model(self, features):
        """
        Use the trained model to make predictions
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            Dictionary with prediction results
        """
        logger.info("Making predictions with trained model...")
        
        # Select and scale features
        X = features[self.selected_features].values
        X_scaled = self.model.scaler.transform(X)
        
        # Generate predictions from base models
        dt_preds = self.model.dt_model.predict(X_scaled).reshape(-1, 1)
        rf_preds = self.model.rf_model.predict(X_scaled).reshape(-1, 1)
        et_preds = self.model.et_model.predict(X_scaled).reshape(-1, 1)
        xgb_preds = self.model.xgb_model.predict(X_scaled).reshape(-1, 1)
        
        # Combine predictions for stacking
        meta_features = np.hstack((dt_preds, rf_preds, et_preds, xgb_preds))
        
        # Make final predictions
        predictions = self.model.stacking_model.predict(meta_features)
        probabilities = self.model.stacking_model.predict_proba(meta_features)
        
        # Get confidence (max probability)
        confidence = np.max(probabilities, axis=1)
        
        # Determine if predictions are attacks or benign
        is_attack = np.array([pred != 'BENIGN' for pred in predictions])
        
        # Package results
        results = {
            'predictions': predictions,
            'confidence': confidence,
            'is_attack': is_attack,
            'base_predictions': {
                'dt': dt_preds.flatten(),
                'rf': rf_preds.flatten(),
                'et': et_preds.flatten(),
                'xgb': xgb_preds.flatten()
            }
        }
        
        return results
    
    def detect_anomalies(self, features):
        """
        Detect anomalies using the anomaly detection model and baseline profile
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            Dictionary with anomaly detection results
        """
        logger.info("Detecting anomalies...")
        
        # Initialize results
        results = {
            'is_anomaly': np.zeros(len(features), dtype=bool),
            'anomaly_score': np.zeros(len(features)),
            'anomaly_details': [[] for _ in range(len(features))]
        }
        
        # If model has KMeans clustering, use it for anomaly detection
        if hasattr(self.model, 'kmeans_model') and self.model.kmeans_model is not None:
            # Apply KMeans clustering
            X = features[self.selected_features].values
            X_scaled = self.model.scaler.transform(X)
            cluster_labels = self.model.kmeans_model.predict(X_scaled)
            
            # Map clusters to classes based on the stored mapping
            predictions = np.array([self.model.cluster_to_class_mapping.get(label, 'BENIGN') 
                                   for label in cluster_labels])
            
            # Identify anomalies (non-BENIGN)
            results['is_anomaly'] = np.array([pred != 'BENIGN' for pred in predictions])
            results['kmeans_predictions'] = predictions
            results['cluster_id'] = cluster_labels
            
            # Apply biased classifiers if available
            if hasattr(self.model, 'biased_classifier_fn') and self.model.biased_classifier_fn is not None:
                # Get potential false negatives
                potential_fn = np.where(~results['is_anomaly'])[0]
                if len(potential_fn) > 0:
                    # Apply FN classifier to improve detection
                    fn_features = X_scaled[potential_fn]
                    fn_preds = self.model.biased_classifier_fn.predict(fn_features)
                    fn_proba = self.model.biased_classifier_fn.predict_proba(fn_features)[:, 1]
                    
                    # Update anomaly results
                    for i, idx in enumerate(potential_fn):
                        if fn_preds[i] == 1:
                            results['is_anomaly'][idx] = True
                            results['anomaly_score'][idx] = fn_proba[i]
            
            # Set anomaly scores for anomalies detected by KMeans
            anomaly_indices = np.where(results['is_anomaly'])[0]
            results['anomaly_score'][anomaly_indices] = 0.8  # Default score for KMeans anomalies
        
        # Use baseline profile for additional anomaly detection
        if self.baseline is not None:
            logger.info("Using baseline profile for anomaly detection...")
            
            # Calculate anomaly scores based on baseline
            for column in features.columns:
                if column in self.baseline and column not in ['Flow ID', 'Source IP', 'Destination IP', 'Protocol']:
                    stats = self.baseline[column]
                    
                    # Calculate Z-score
                    if stats['std'] > 0:
                        z_scores = (features[column] - stats['mean']) / stats['std']
                    else:
                        z_scores = np.zeros(len(features))
                    
                    # Identify outliers based on IQR
                    iqr = stats['iqr']
                    if iqr > 0:
                        lower_bound = stats['q1'] - 1.5 * iqr
                        upper_bound = stats['q3'] + 1.5 * iqr
                        outliers = ((features[column] < lower_bound) | (features[column] > upper_bound))
                    else:
                        outliers = np.zeros(len(features), dtype=bool)
                    
                    # Accumulate anomaly scores
                    results['anomaly_score'] += np.abs(z_scores) / len(self.baseline)
                    
                    # Store details for significant anomalies
                    for i in range(len(features)):
                        z_score = z_scores.iloc[i] if hasattr(z_scores, 'iloc') else z_scores[i]
                        is_outlier = outliers.iloc[i] if hasattr(outliers, 'iloc') else outliers[i]
                        
                        if abs(z_score) > 3 or is_outlier:
                            results['anomaly_details'][i].append({
                                'feature': column,
                                'value': float(features.iloc[i][column]),
                                'z_score': float(z_score),
                                'is_outlier': bool(is_outlier),
                                'baseline_mean': float(stats['mean']),
                                'baseline_std': float(stats['std'])
                            })
            
            # Identify anomalies based on anomaly score threshold
            baseline_anomalies = results['anomaly_score'] > 0.7
            
            # Combine with existing anomalies
            results['is_anomaly'] = results['is_anomaly'] | baseline_anomalies
        
        return results
    
    def analyze_records(self, features, original_df):
        """
        Analyze records using both model predictions and anomaly detection
        
        Args:
            features: DataFrame with extracted features
            original_df: Original DataFrame with eve.json records
        """
        logger.info("=" * 60)
        logger.info(f"Analyzing {len(features)} records...")
        
        # Make predictions with the trained model
        ml_results = self.predict_with_model(features)
        
        # Detect anomalies
        anomaly_results = self.detect_anomalies(features)
        
        # Combine results - records are flagged if either method detects an issue
        is_flagged = ml_results['is_attack'] | anomaly_results['is_anomaly']
        
        # Update alert count
        self.stats['alerts'] += np.sum(is_flagged)
        
        # Generate alerts for flagged records
        if np.any(is_flagged):
            self.generate_alerts(is_flagged, ml_results, anomaly_results, features, original_df)
        else:
            logger.info("No threats detected in this batch")
        
        # Print summary
        self.print_summary()
    
    def generate_alerts(self, is_flagged, ml_results, anomaly_results, features, original_df):
        """
        Generate alerts for flagged records
        
        Args:
            is_flagged: Boolean array indicating flagged records
            ml_results: Results from ML prediction
            anomaly_results: Results from anomaly detection
            features: Extracted features
            original_df: Original DataFrame with eve.json records
        """
        # Get indices of flagged records
        flagged_indices = np.where(is_flagged)[0]
        
        logger.info("=" * 60)
        logger.info(f"ALERT REPORT: {len(flagged_indices)} potential threats detected")
        logger.info("=" * 60)
        
        # Process each flagged record
        for i, idx in enumerate(flagged_indices, 1):
            # Extract information about the record
            record = {}
            for field in ['timestamp', 'flow_id', 'src_ip', 'src_port', 'dest_ip', 'dest_port', 'proto', 'event_type']:
                if field in original_df.columns:
                    record[field] = original_df.iloc[idx][field]
            
            # Determine reason for flagging
            reasons = []
            if ml_results['is_attack'][idx]:
                reasons.append(f"ML Classification: {ml_results['predictions'][idx]} " +
                             f"(Confidence: {ml_results['confidence'][idx]:.2f})")
            
            if anomaly_results['is_anomaly'][idx]:
                reasons.append(f"Anomaly Detection: Score {anomaly_results['anomaly_score'][idx]:.2f}")
                
                # Add cluster information if available
                if 'cluster_id' in anomaly_results:
                    cluster_id = anomaly_results['cluster_id'][idx]
                    predicted_class = anomaly_results.get('kmeans_predictions', [''])[idx]
                    reasons.append(f"Cluster Analysis: Cluster {cluster_id} -> {predicted_class}")
            
            # Print alert
            logger.info(f"\nALERT #{i}:")
            logger.info(f"Timestamp: {record.get('timestamp', 'N/A')}")
            logger.info(f"Flow ID: {record.get('flow_id', 'N/A')}")
            logger.info(f"Source: {record.get('src_ip', 'N/A')}:{record.get('src_port', 'N/A')}")
            logger.info(f"Destination: {record.get('dest_ip', 'N/A')}:{record.get('dest_port', 'N/A')}")
            logger.info(f"Protocol: {record.get('proto', 'N/A')}")
            logger.info(f"Event Type: {record.get('event_type', 'N/A')}")
            
            logger.info("\nDetection Reasons:")
            for reason in reasons:
                logger.info(f"- {reason}")
            
            # Print anomaly details if available
            if anomaly_results['anomaly_details'][idx]:
                logger.info("\nAnomaly Details:")
                # Sort by absolute z-score
                details = sorted(anomaly_results['anomaly_details'][idx], 
                               key=lambda x: abs(x['z_score']), reverse=True)
                
                for j, detail in enumerate(details[:5], 1):  # Show top 5 anomalies
                    logger.info(f"  {j}. {detail['feature']}: {detail['value']:.2f} " +
                              f"(Z-score: {detail['z_score']:.2f}, Baseline: {detail['baseline_mean']:.2f}±{detail['baseline_std']:.2f})")
            
            # Print flow information if available
            if 'flow' in original_df.columns and isinstance(original_df.iloc[idx]['flow'], dict):
                flow = original_df.iloc[idx]['flow']
                logger.info("\nFlow Information:")
                for key, value in flow.items():
                    logger.info(f"  {key}: {value}")
            
            # Print TCP information if available
            if 'tcp' in original_df.columns and isinstance(original_df.iloc[idx]['tcp'], dict):
                tcp = original_df.iloc[idx]['tcp']
                logger.info("\nTCP Information:")
                for key, value in tcp.items():
                    logger.info(f"  {key}: {value}")
            
            # Print HTTP information if available
            if 'http' in original_df.columns and isinstance(original_df.iloc[idx]['http'], dict):
                http = original_df.iloc[idx]['http']
                logger.info("\nHTTP Information:")
                for key, value in http.items():
                    logger.info(f"  {key}: {value}")
            
            # Print DNS information if available
            if 'dns' in original_df.columns and isinstance(original_df.iloc[idx]['dns'], dict):
                dns = original_df.iloc[idx]['dns']
                logger.info("\nDNS Information:")
                for key, value in dns.items():
                    if isinstance(value, (list, dict)):
                        logger.info(f"  {key}: {json.dumps(value, indent=2)}")
                    else:
                        logger.info(f"  {key}: {value}")
            
            # Add separator between alerts
            logger.info("-" * 60)
    
    def print_summary(self):
        """
        Print a summary of the detection statistics
        """
        duration = (datetime.datetime.now() - self.stats['start_time']).total_seconds()
        
        logger.info("=" * 60)
        logger.info("DETECTION SUMMARY")
        logger.info("-" * 60)
        logger.info(f"Records processed: {self.stats['processed']}")
        logger.info(f"Alerts generated: {self.stats['alerts']}")
        
        if self.stats['processed'] > 0:
            alert_rate = self.stats['alerts'] / self.stats['processed'] * 100
            logger.info(f"Alert rate: {alert_rate:.2f}%")
        
        if duration > 0:
            records_per_second = self.stats['processed'] / duration
            logger.info(f"Processing rate: {records_per_second:.2f} records/second")
        
        if self.is_monitoring:
            logger.info(f"Monitoring duration: {duration:.1f} seconds")
        
        logger.info("=" * 60)
    
    def analyze_file(self):
        """
        Analyze the entire eve.json file
        """
        logger.info(f"Starting analysis of {self.eve_json_path}")
        
        # Process the file
        df = self.process_eve_json()
        
        if len(df) == 0:
            logger.warning("No valid records found in the file")
            return
        
        # Extract features
        features = self.extract_features(df)
        
        # Analyze records
        self.analyze_records(features, df)
        
        logger.info("Analysis completed")
    
    def start_monitoring(self):
        """
        Start continuous monitoring of the eve.json file
        """
        logger.info(f"Starting continuous monitoring of {self.eve_json_path}")
        logger.info("Press Ctrl+C to stop monitoring")
        
        # Set initial position to end of file
        try:
            with open(self.eve_json_path, 'r') as f:
                f.seek(0, 2)  # Seek to end
                self.last_position = f.tell()
                logger.info(f"Initial file position: {self.last_position}")
        except Exception as e:
            logger.error(f"Error initializing file position: {e}")
            return
        
        # Set up signal handler for graceful termination
        def signal_handler(sig, frame):
            logger.info("\nMonitoring stopped")
            self.print_summary()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Monitoring loop
        try:
            while True:
                # Check for new entries
                self.process_new_entries()
                time.sleep(1)  # Check every second
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
            self.print_summary()


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NIDS Detection Module')
    parser.add_argument('--mode', choices=['analyze', 'monitor'], required=True,
                      help='Operation mode: analyze (single file) or monitor (continuous)')
    parser.add_argument('--model', required=True, help='Path to the trained model directory')
    parser.add_argument('--evejson', required=True, help='Path to the Suricata eve.json file')
    parser.add_argument('--threshold', type=float, default=0.8, help='Detection threshold (0.0-1.0)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Create detector
        detector = NIDS_Detector(
            model_dir=args.model,
            eve_json_path=args.evejson,
            detection_threshold=args.threshold,
            monitoring=(args.mode == 'monitor')
        )
        
        # Execute the selected mode
        if args.mode == 'analyze':
            detector.analyze_file()
        elif args.mode == 'monitor':
            detector.start_monitoring()
        else:
            logger.error(f"Unknown mode: {args.mode}")
    
    except KeyboardInterrupt:
        logger.info("\nOperation interrupted by user")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.debug(traceback.format_exc())


if __name__ == '__main__':
    main()
