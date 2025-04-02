#!/usr/bin/env python3
"""
Network Intrusion Detection System - Detection Module

This script implements a hybrid NIDS that can analyze static Eve JSON files or
monitor in real-time to detect potential zero-day attacks. It uses both a trained 
classification model and a statistical baseline model for detection.

Based on the paper "A Suricata and Machine Learning Based Hybrid Network Intrusion Detection System"
"""

import argparse
import os
import sys
import time
import json
import pickle
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

class EveJsonHandler(PatternMatchingEventHandler):
    """
    File handler for monitoring changes to Eve JSON file in real-time.
    """
    def __init__(self, detector, *args, **kwargs):
        super(EveJsonHandler, self).__init__(*args, **kwargs)
        self.detector = detector
        self.last_position = 0

    def on_modified(self, event):
        """
        Called when the watched file is modified.
        """
        self.detector.process_new_entries()


class IntrusionDetector:
    """
    Main class for handling intrusion detection using both classification and baseline models.
    """
    def __init__(self, model_dir, eve_json_path=None, is_monitoring=False):
        """
        Initialize the detector with the model directory and optional Eve JSON path.
        
        Args:
            model_dir: Directory containing the trained model and baseline
            eve_json_path: Path to the Eve JSON file (optional)
            is_monitoring: Boolean indicating if we're in monitoring mode
        """
        self.model_dir = model_dir
        self.eve_json_path = eve_json_path
        self.is_monitoring = is_monitoring
        
        # Load models and configuration
        self.load_models()
        
        # Initialize statistics
        self.stats = {
            'processed': 0,
            'alerts_classification': 0,
            'alerts_baseline': 0,
            'alerts_combined': 0,
            'start_time': datetime.datetime.now()
        }
        
        # Set file position to end if monitoring
        if is_monitoring and eve_json_path:
            with open(eve_json_path, 'r') as f:
                f.seek(0, 2)  # Go to the end of the file
                self.last_position = f.tell()
    
    def load_models(self):
        """
        Load the classification model, selected features, and baseline profile.
        """
        print("Loading models and configuration...")
        
        # Load classification model
        model_path = os.path.join(self.model_dir, 'model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.clf_model = pickle.load(f)
            print("Classification model loaded successfully.")
        else:
            print(f"Warning: Classification model not found at {model_path}")
            self.clf_model = None
        
        # Load selected features
        features_path = os.path.join(self.model_dir, 'selected_features.json')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.selected_features = json.load(f)
            print(f"Loaded {len(self.selected_features)} selected features.")
        else:
            print(f"Warning: Selected features not found at {features_path}")
            self.selected_features = []
        
        # Load baseline profile
        baseline_path = os.path.join(self.model_dir, 'baseline.json')
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                self.baseline = json.load(f)
            print("Baseline profile loaded successfully.")
        else:
            print(f"Warning: Baseline profile not found at {baseline_path}")
            self.baseline = None
        
        # Load metadata
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Model type: {self.metadata.get('model_type', 'unknown')}")
        else:
            print(f"Warning: Metadata not found at {metadata_path}")
            self.metadata = {}
    
    def process_eve_json(self):
        """_
        Process the entire Eve JSON file for analysis.
        
        Returns:
            DataFrame of extracted and processed records
        """
        print(f"Processing Eve JSON file: {self.eve_json_path}")
        
        # Read the file line by line as each line is a separate JSON object
        data = []
        with open(self.eve_json_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Skip records with event_type = "alert" (existing Suricata alerts)
                    if record.get('event_type') == 'alert':
                        continue
                    data.append(record)
                except json.JSONDecodeError:
                    print(f"Error parsing JSON line: {line[:50]}...")
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} records from Eve JSON file (excluding alerts)")
        
        return df
    
    def process_new_entries(self):
        """
        Process new entries added to the Eve JSON file since last check.
        Used for real-time monitoring.
        """
        if not self.eve_json_path:
            print("Error: No Eve JSON file specified for monitoring")
            return
        
        # Open file and move to last position
        with open(self.eve_json_path, 'r') as f:
            f.seek(self.last_position)
            
            # Read and process new lines
            new_records = []
            for line in f:
                try:
                    record = json.loads(line)
                    # Skip records with event_type = "alert" (existing Suricata alerts)
                    if record.get('event_type') == 'alert':
                        continue
                    new_records.append(record)
                except json.JSONDecodeError:
                    print(f"Error parsing JSON line: {line[:50]}...")
                    continue
            
            # Update last position
            self.last_position = f.tell()
        
        # Process the new records if any
        if new_records:
            df = pd.DataFrame(new_records)
            print(f"Processing {len(df)} new records...")
            
            # Process and analyze
            features = self.extract_features(df)
            self.analyze_flows(features, df)
    
    def extract_features(self, df):
        """
        Extract and transform features from Eve JSON records to match model features.
        
        Args:
            df: DataFrame with parsed Eve JSON records
            
        Returns:
            DataFrame with extracted features matching the model's expected features
        """
        # Initialize an empty DataFrame to store the extracted features
        features = pd.DataFrame(index=df.index)
        
        # Feature mapping between eve.json and CICIDS2017
        # This is a simplified mapping - expand based on your actual data
        feature_mapping = {
            'Flow ID': 'flow_id',
            'Source IP': 'src_ip',
            'Source Port': 'src_port',
            'Destination IP': 'dest_ip',
            'Destination Port': 'dest_port',
            'Protocol': 'proto'
        }
        
        # Map direct fields
        for target_field, source_field in feature_mapping.items():
            if source_field in df.columns:
                features[target_field] = df[source_field]
            else:
                features[target_field] = None
        
        # Extract flow information
        for idx, row in df.iterrows():
            # Parse timestamp if available
            if 'timestamp' in row:
                try:
                    # Convert timestamp to datetime
                    timestamp = pd.to_datetime(row['timestamp'])
                    # Extract components
                    features.at[idx, 'Year'] = timestamp.year
                    features.at[idx, 'Month'] = timestamp.month
                    features.at[idx, 'Day'] = timestamp.day
                    features.at[idx, 'Hour'] = timestamp.hour
                    features.at[idx, 'Minute'] = timestamp.minute
                    features.at[idx, 'Second'] = timestamp.second
                except (ValueError, TypeError):
                    # Default values if parsing fails
                    features.at[idx, 'Year'] = 2023
                    features.at[idx, 'Month'] = 1
                    features.at[idx, 'Day'] = 1
                    features.at[idx, 'Hour'] = 0
                    features.at[idx, 'Minute'] = 0
                    features.at[idx, 'Second'] = 0
            
            # Check if 'flow' is in the record
            if 'flow' in row and isinstance(row['flow'], dict):
                flow = row['flow']
                features.at[idx, 'Flow Duration'] = flow.get('age', 0)
                features.at[idx, 'Total Fwd Packets'] = flow.get('pkts_toserver', 0)
                features.at[idx, 'Total Backward Packets'] = flow.get('pkts_toclient', 0)
                features.at[idx, 'Total Length of Fwd Packets'] = flow.get('bytes_toserver', 0)
                features.at[idx, 'Total Length of Bwd Packets'] = flow.get('bytes_toclient', 0)
                
                
                total_packets = flow.get('pkts_toserver', 0) + flow.get('pkts_toclient', 0)
                total_bytes = flow.get('bytes_toserver', 0) + flow.get('bytes_toclient', 0)
                
                # Handle missing features
                if total_packets > 0:
                    features.at[idx, 'Average Packet Size'] = total_bytes / total_packets
                else:
                    features.at[idx, 'Average Packet Size'] = 0
                    
                # Estimate packet length statistics
                if 'bytes_toserver' in flow and flow.get('pkts_toserver', 0) > 0:
                    avg_pkt_size_fwd = flow['bytes_toserver'] / flow['pkts_toserver']
                else:
                    avg_pkt_size_fwd = 0
                    
                if 'bytes_toclient' in flow and flow.get('pkts_toclient', 0) > 0:
                    features.at[idx, 'Avg Bwd Segment Size'] = flow['bytes_toclient'] / flow['pkts_toclient']
                    features.at[idx, 'Bwd Packet Length Mean'] = features.at[idx, 'Avg Bwd Segment Size']
                    features.at[idx, 'Bwd Packet Length Max'] = features.at[idx, 'Avg Bwd Segment Size'] * 2  # Estimation
                else:
                    features.at[idx, 'Avg Bwd Segment Size'] = 0
                    features.at[idx, 'Bwd Packet Length Mean'] = 0
                    features.at[idx, 'Bwd Packet Length Max'] = 0
                
                # Estimate packet length variance and std (using a heuristic)
                features.at[idx, 'Packet Length Mean'] = total_bytes / total_packets if total_packets > 0 else 0
                features.at[idx, 'Packet Length Variance'] = (avg_pkt_size_fwd - features.at[idx, 'Avg Bwd Segment Size'])**2
                features.at[idx, 'Packet Length Std'] = np.sqrt(features.at[idx, 'Packet Length Variance'])
                features.at[idx, 'Bwd Packet Length Std'] = features.at[idx, 'Packet Length Std'] / 2  # Estimation
                
                # Estimate max packet length
                features.at[idx, 'Max Packet Length'] = max(avg_pkt_size_fwd * 2, features.at[idx, 'Avg Bwd Segment Size'] * 2)
                
                # Subflow features - assumed same as flow in this context
                features.at[idx, 'Subflow Fwd Packets'] = flow.get('pkts_toserver', 0)
                features.at[idx, 'Subflow Fwd Bytes'] = flow.get('bytes_toserver', 0)
                features.at[idx, 'Subflow Bwd Bytes'] = flow.get('bytes_toclient', 0)
                
                # Header length estimation (typical TCP/IP header is ~40 bytes)
                header_size = 40  # Typical TCP/IP header size
                features.at[idx, 'Fwd Header Length'] = header_size * flow.get('pkts_toserver', 0)
                features.at[idx, 'Fwd Header Length.1'] = features.at[idx, 'Fwd Header Length']  # Duplicate
                
                # Estimate active data packets
                features.at[idx, 'act_data_pkt_fwd'] = flow.get('pkts_toserver', 0)
                
                # Flow IAT (Inter-Arrival Time) statistics 
                # These are harder to estimate from aggregated flow data
                duration = flow.get('age', 0)
                if duration > 0 and total_packets > 2:
                    avg_iat = duration / (total_packets - 1)
                    features.at[idx, 'Flow IAT Std'] = avg_iat / 2  # Rough estimation
                    features.at[idx, 'Fwd IAT Max'] = avg_iat * 3  # Rough estimation
                    features.at[idx, 'Fwd IAT Std'] = avg_iat / 2  # Rough estimation
                else:
                    features.at[idx, 'Flow IAT Std'] = 0
                    features.at[idx, 'Fwd IAT Max'] = 0
                    features.at[idx, 'Fwd IAT Std'] = 0
                    
                # Calculate packet and byte rates if duration is available
                if features.at[idx, 'Flow Duration'] > 0:
                    features.at[idx, 'Flow Bytes/s'] = (features.at[idx, 'Total Length of Fwd Packets'] + 
                                                       features.at[idx, 'Total Length of Bwd Packets']) / features.at[idx, 'Flow Duration']
                    features.at[idx, 'Flow Packets/s'] = (features.at[idx, 'Total Fwd Packets'] + 
                                                         features.at[idx, 'Total Backward Packets']) / features.at[idx, 'Flow Duration']
                    features.at[idx, 'Fwd Packets/s'] = features.at[idx, 'Total Fwd Packets'] / features.at[idx, 'Flow Duration']
                    features.at[idx, 'Bwd Packets/s'] = features.at[idx, 'Total Backward Packets'] / features.at[idx, 'Flow Duration']
                else:
                    features.at[idx, 'Flow Bytes/s'] = 0
                    features.at[idx, 'Flow Packets/s'] = 0
                    features.at[idx, 'Fwd Packets/s'] = 0
                    features.at[idx, 'Bwd Packets/s'] = 0
            else:
                # Default values if flow information is not available
                features.at[idx, 'Flow Duration'] = 0
                features.at[idx, 'Total Fwd Packets'] = 0
                features.at[idx, 'Total Backward Packets'] = 0
                features.at[idx, 'Total Length of Fwd Packets'] = 0
                features.at[idx, 'Total Length of Bwd Packets'] = 0
                features.at[idx, 'Flow Bytes/s'] = 0
                features.at[idx, 'Flow Packets/s'] = 0
                features.at[idx, 'Fwd Packets/s'] = 0
                features.at[idx, 'Bwd Packets/s'] = 0
            
            # Extract TCP information if available
            if 'tcp' in row and isinstance(row['tcp'], dict):
                tcp = row['tcp']
                # Extract TCP flags
                features.at[idx, 'FIN Flag Count'] = 1 if tcp.get('fin', False) else 0
                features.at[idx, 'SYN Flag Count'] = 1 if tcp.get('syn', False) else 0
                features.at[idx, 'RST Flag Count'] = 1 if tcp.get('rst', False) else 0
                features.at[idx, 'PSH Flag Count'] = 1 if tcp.get('psh', False) else 0
                features.at[idx, 'ACK Flag Count'] = 1 if tcp.get('ack', False) else 0
                features.at[idx, 'URG Flag Count'] = 1 if tcp.get('urg', False) else 0
                
                # Extract window size if available
                if 'tcp_options' in tcp and isinstance(tcp['tcp_options'], list):
                    for option in tcp['tcp_options']:
                        if option.get('type') == 'window scale':
                            features.at[idx, 'Init_Win_bytes_forward'] = option.get('value', 0)
                            break
            else:
                # Default values if TCP information is not available
                features.at[idx, 'FIN Flag Count'] = 0
                features.at[idx, 'SYN Flag Count'] = 0
                features.at[idx, 'RST Flag Count'] = 0
                features.at[idx, 'PSH Flag Count'] = 0
                features.at[idx, 'ACK Flag Count'] = 0
                features.at[idx, 'URG Flag Count'] = 0
                features.at[idx, 'Init_Win_bytes_forward'] = 0
            
            # Add more feature extraction logic here based on the available data in eve.json
            # and the required features for the model
        
        # Fill any missing values with zeros
        features.fillna(0, inplace=True)
        
        # Select only the features needed by the model if available
        if self.selected_features and all(f in features.columns for f in self.selected_features):
            features = features[self.selected_features]
        
        return features
    
    def analyze_classification(self, features):
        """
        Use the classification model to classify the flows.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            DataFrame with classification results
        """
        if self.clf_model is None:
            print("Error: Classification model not loaded")
            return pd.DataFrame()
        
        print("Analyzing flows using classification model...")
        
        # Ensure all required features are present
        missing_features = [f for f in self.selected_features if f not in features.columns]
        if missing_features:
            print(f"Warning: Missing features for classification: {missing_features}")
            for feature in missing_features:
                features[feature] = 0  # Add missing features with default value of 0
        
        # Reorder columns to match the model's expected order
        features = features[self.selected_features]
        
        # Make predictions
        if hasattr(self.clf_model, 'predict_proba'):
            # Get probability predictions
            if hasattr(self.clf_model, 'label_encoder_'):
                # XGBoost model
                y_proba = self.clf_model.predict_proba(features)
                # Get the maximum probability for each prediction
                max_proba = np.max(y_proba, axis=1)
                # Get predicted class
                y_pred = self.clf_model.predict(features)
                # Map predictions back to class names
                y_pred_labels = self.clf_model.label_encoder_.inverse_transform(y_pred)
            else:
                # Random Forest model
                y_proba = self.clf_model.predict_proba(features)
                # Get the maximum probability for each prediction
                max_proba = np.max(y_proba, axis=1)
                # Get predicted class
                y_pred_labels = self.clf_model.predict(features)
        else:
            # Fallback if predict_proba is not available
            y_pred_labels = self.clf_model.predict(features)
            max_proba = np.ones(len(features))  # Default to 1.0 confidence
        
        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': y_pred_labels,
            'confidence': max_proba,
            'is_attack': np.where(y_pred_labels == 'BENIGN', False, True)
        })
        
        return results
    
    def analyze_baseline(self, features):
        """
        Use the baseline model to detect anomalies.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            DataFrame with anomaly detection results
        """
        if self.baseline is None:
            print("Error: Baseline profile not loaded")
            return pd.DataFrame()
        
        print("Analyzing flows using baseline model...")
        
        # Initialize results
        anomaly_scores = np.zeros(len(features))
        anomalies = {}
        
        # Check each feature against the baseline
        for column in features.columns:
            if column in ['Flow ID', 'Source IP', 'Destination IP', 'Protocol'] or column not in self.baseline:
                continue
            
            # Get baseline stats for this feature
            stats = self.baseline[column]
            
            # Calculate Z-score: (x - mean) / std
            if stats['std'] > 0:
                z_scores = (features[column] - stats['mean']) / stats['std']
            else:
                z_scores = np.zeros(len(features))
            
            # Calculate outlier score based on IQR
            iqr = stats['iqr']
            if iqr > 0:
                lower_bound = stats['q1'] - 1.5 * iqr
                upper_bound = stats['q3'] + 1.5 * iqr
                # Count values outside the bounds
                outliers = ((features[column] < lower_bound) | (features[column] > upper_bound))
            else:
                outliers = np.zeros(len(features), dtype=bool)
            
            # Accumulate the absolute z-scores
            anomaly_scores += np.abs(z_scores)
            
            # Store specific anomalies for reporting
            # To this:
            z_vals = z_scores.values if hasattr(z_scores, 'values') else z_scores
            o_vals = outliers.values if hasattr(outliers, 'values') else outliers
            for i, (z, is_outlier) in enumerate(zip(z_vals, o_vals)):
                if abs(z) > 3 or is_outlier:  # Threshold can be adjusted
                    if i not in anomalies:
                        anomalies[i] = []
                    anomalies[i].append({
                        'feature': column,
                        'value': features.iloc[i][column],
                        'z_score': z,
                        'is_outlier': is_outlier,
                        'baseline_mean': stats['mean'],
                        'baseline_std': stats['std'],
                        'baseline_q1': stats['q1'],
                        'baseline_q3': stats['q3']
                    })
        
        # Normalize anomaly scores
        if len(features.columns) > 0:
            anomaly_scores /= len(features.columns)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'anomaly_score': anomaly_scores,
            'is_anomaly': anomaly_scores > 0.5,  # Threshold can be adjusted
            'anomaly_details': [anomalies.get(i, []) for i in range(len(features))]
        })
        
        return results
    
    def analyze_flows(self, features, original_df):
        """
        Analyze network flows using both classification and baseline models.
        
        Args:
            features: DataFrame with extracted features
            original_df: Original DataFrame with Eve JSON records
        """
        print(f"Analyzing {len(features)} flows...")
        
        # Update statistics
        self.stats['processed'] += len(features)
        
        # Run classification analysis
        if self.clf_model is not None:
            clf_results = self.analyze_classification(features)
            self.stats['alerts_classification'] += clf_results['is_attack'].sum()
        else:
            clf_results = pd.DataFrame(index=features.index)
            clf_results['is_attack'] = False
            clf_results['prediction'] = 'UNKNOWN'
            clf_results['confidence'] = 0.0
        
        # Run baseline analysis
        if self.baseline is not None:
            baseline_results = self.analyze_baseline(features)
            self.stats['alerts_baseline'] += baseline_results['is_anomaly'].sum()
        else:
            baseline_results = pd.DataFrame(index=features.index)
            baseline_results['is_anomaly'] = False
            baseline_results['anomaly_score'] = 0.0
            baseline_results['anomaly_details'] = [[] for _ in range(len(features))]
        
        # Combine results - a flow is flagged if either model detects an issue
        combined_results = pd.DataFrame(index=features.index)
        combined_results['is_alert'] = clf_results['is_attack'] | baseline_results['is_anomaly']
        combined_results['classification_prediction'] = clf_results['prediction']
        combined_results['classification_confidence'] = clf_results['confidence']
        combined_results['baseline_anomaly_score'] = baseline_results['anomaly_score']
        combined_results['baseline_anomaly_details'] = baseline_results['anomaly_details']
        
        # Update combined alerts count
        self.stats['alerts_combined'] += combined_results['is_alert'].sum()
        
        # Generate report for detected alerts
        self.report_alerts(combined_results, original_df)
        
        # Print summary
        self.print_summary()
    
    def report_alerts(self, results, original_df):
        """
        Generate and print a report for detected alerts.
        
        Args:
            results: DataFrame with combined detection results
            original_df: Original DataFrame with Eve JSON records
        """
        # Filter alerts
        alerts = results[results['is_alert']].copy()
        
        if len(alerts) == 0:
            print("No alerts detected in this batch.")
            return
        
        print("\n" + "="*80)
        print(f"ALERT REPORT: {len(alerts)} potential intrusions detected")
        print("="*80)
        
        # Sort alerts by anomaly score and confidence
        alerts['combined_score'] = alerts['classification_confidence'] + alerts['baseline_anomaly_score']
        alerts = alerts.sort_values('combined_score', ascending=False)
        
        # Print each alert
        for i, (idx, alert) in enumerate(alerts.iterrows(), 1):
            # Get original record
            original = original_df.iloc[idx]
            
            print(f"\nALERT #{i}:")
            print(f"Timestamp: {original.get('timestamp', 'N/A')}")
            print(f"Source: {original.get('src_ip', 'N/A')}:{original.get('src_port', 'N/A')}")
            print(f"Destination: {original.get('dest_ip', 'N/A')}:{original.get('dest_port', 'N/A')}")
            print(f"Protocol: {original.get('proto', 'N/A')}")
            print(f"Event Type: {original.get('event_type', 'N/A')}")
            
            # Print classification results
            if alert['classification_confidence'] > 0:
                print(f"Classification: {alert['classification_prediction']} "
                      f"(Confidence: {alert['classification_confidence']:.4f})")
            
            # Print baseline anomaly results
            if alert['baseline_anomaly_score'] > 0:
                print(f"Baseline Anomaly Score: {alert['baseline_anomaly_score']:.4f}")
                
                # Print top anomalies
                if len(alert['baseline_anomaly_details']) > 0:
                    print("Top Anomalies:")
                    # Sort anomalies by z-score
                    anomalies = sorted(alert['baseline_anomaly_details'], 
                                     key=lambda x: abs(x['z_score']), reverse=True)
                    for j, anomaly in enumerate(anomalies[:3], 1):  # Print top 3 anomalies
                        print(f"  {j}. {anomaly['feature']}: {anomaly['value']} "
                              f"(Z-score: {anomaly['z_score']:.2f}, "
                              f"Baseline: {anomaly['baseline_mean']:.2f}±{anomaly['baseline_std']:.2f})")
            
            # Print additional flow information if available
            if 'flow' in original and isinstance(original['flow'], dict):
                flow = original['flow']
                print("Flow Information:")
                print(f"  Duration: {flow.get('age', 'N/A')} seconds")
                print(f"  Packets: {flow.get('pkts_toserver', 0)} to server, "
                      f"{flow.get('pkts_toclient', 0)} to client")
                print(f"  Bytes: {flow.get('bytes_toserver', 0)} to server, "
                      f"{flow.get('bytes_toclient', 0)} to client")
            
            # Print original json for debugging if needed
            # print("Original JSON:", json.dumps(original.to_dict(), indent=2))
        
        print("\n" + "="*80)
    
    def print_summary(self):
        """
        Print a summary of the detection statistics.
        """
        duration = (datetime.datetime.now() - self.stats['start_time']).total_seconds()
        
        print("\n" + "-"*40)
        print("DETECTION SUMMARY")
        print("-"*40)
        print(f"Total flows processed: {self.stats['processed']}")
        print(f"Classification alerts: {self.stats['alerts_classification']}")
        print(f"Baseline anomaly alerts: {self.stats['alerts_baseline']}")
        print(f"Combined alerts: {self.stats['alerts_combined']}")
        
        if duration > 0:
            print(f"Processing rate: {self.stats['processed'] / duration:.2f} flows/second")
        
        if self.is_monitoring:
            print(f"Monitoring for: {duration:.1f} seconds")
        
        print("-"*40 + "\n")
    
    def analyze_file(self):
        """
        Analyze a static Eve JSON file for potential intrusions.
        """
        if not self.eve_json_path:
            print("Error: No Eve JSON file specified for analysis")
            return
        
        # Process the file
        df = self.process_eve_json()
        
        # Extract features
        features = self.extract_features(df)
        
        # Analyze flows
        self.analyze_flows(features, df)
    
    def start_monitoring(self):
        """
        Start continuous monitoring of the Eve JSON file.
        """
        if not self.eve_json_path:
            print("Error: No Eve JSON file specified for monitoring")
            return
        
        print(f"Starting continuous monitoring of {self.eve_json_path}...")
        print("Press Ctrl+C to stop monitoring.")
        
        # Set up file observer
        event_handler = EveJsonHandler(
            self,
            patterns=[os.path.basename(self.eve_json_path)],
            ignore_directories=True
        )
        
        observer = Observer()
        observer.schedule(event_handler, os.path.dirname(self.eve_json_path), recursive=False)
        observer.start()
        
        try:
            # Keep running until interrupted
            while True:
                # Check for new entries periodically
                self.process_new_entries()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            observer.stop()
        
        observer.join()


def main():
    """
    Main function to parse arguments and execute the selected mode.
    """
    parser = argparse.ArgumentParser(description='Network Intrusion Detection System - Detection Module')
    parser.add_argument('--mode', choices=['analyze', 'monitor'], required=True, 
                        help='Operating mode: analyze (for static file analysis) or monitor (for real-time monitoring)')
    parser.add_argument('--model', required=True, help='Path to model directory')
    parser.add_argument('--evejson', required=True, help='Path to eve.json file')
    parser.add_argument('--interface', help='Network interface for monitoring (optional)')
    
    args = parser.parse_args()
    
    # Create detector
    detector = IntrusionDetector(
        model_dir=args.model,
        eve_json_path=args.evejson,
        is_monitoring=(args.mode == 'monitor')
    )
    
    # Execute the selected mode
    if args.mode == 'analyze':
        detector.analyze_file()
    elif args.mode == 'monitor':
        detector.start_monitoring()
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()