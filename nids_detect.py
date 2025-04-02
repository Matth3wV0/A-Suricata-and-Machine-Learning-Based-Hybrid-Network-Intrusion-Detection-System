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
        """
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
                    # Parse the JSON object
                    record = json.loads(line)
                    
                    # Extract the Suricata event data from the nested structure
                    if 'suricata' in record and 'eve' in record['suricata']:
                        event_data = record['suricata']['eve']
                        
                        # Skip records with event_type = "alert" (existing Suricata alerts)
                        if event_data.get('event_type') == 'alert':
                            continue
                        
                        data.append(event_data)
                    else:
                        # Try the old format as fallback
                        if 'event_type' in record and record.get('event_type') != 'alert':
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
                    # Parse the JSON object
                    record = json.loads(line)
                    
                    # Extract the Suricata event data from the nested structure
                    if 'suricata' in record and 'eve' in record['suricata']:
                        event_data = record['suricata']['eve']
                        
                        # Skip records with event_type = "alert" (existing Suricata alerts)
                        if event_data.get('event_type') == 'alert':
                            continue
                        
                        new_records.append(event_data)
                    else:
                        # Try the old format as fallback
                        if 'event_type' in record and record.get('event_type') != 'alert':
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
            
            # Create placeholder for all the features in selected_features
            # to ensure feature order matches the training data
            for feature in self.selected_features:
                features[feature] = 0
            
            float_columns = [
        'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 
        'Fwd Packets/s', 'Bwd Packets/s', 'Average Packet Size',
        'Avg Bwd Segment Size', 'Bwd Packet Length Mean', 'Bwd Packet Length Max',
        'Bwd Packet Length Std', 'Packet Length Mean', 'Packet Length Variance',
        'Packet Length Std', 'Max Packet Length', 'Flow IAT Std',
        'Fwd IAT Std', 'Fwd IAT Max', 'act_data_pkt_fwd'
    ]
            # Initialize float columns with dtype=float64
            for col in float_columns:
                features[col] = pd.Series(0.0, index=df.index, dtype='float64')
            
            
            # Feature mapping between eve.json and CICIDS2017
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
                if source_field in df.columns and target_field in features.columns:
                    features[target_field] = df[source_field]
            
            # Extract flow information
            for idx, row in df.iterrows():
                # Parse timestamp if available
                if 'timestamp' in row:
                    try:
                        # Convert timestamp to datetime
                        timestamp = pd.to_datetime(row['timestamp'])
                        # Extract components if needed for the model
                        if 'Year' in features.columns:
                            features.at[idx, 'Year'] = timestamp.year
                        if 'Month' in features.columns:
                            features.at[idx, 'Month'] = timestamp.month
                        if 'Day' in features.columns:
                            features.at[idx, 'Day'] = timestamp.day
                        if 'Hour' in features.columns:
                            features.at[idx, 'Hour'] = timestamp.hour
                        if 'Minute' in features.columns:
                            features.at[idx, 'Minute'] = timestamp.minute
                        if 'Second' in features.columns:
                            features.at[idx, 'Second'] = timestamp.second
                    except (ValueError, TypeError):
                        pass
                
                # Check if 'flow' is in the record
                if 'flow' in row and isinstance(row['flow'], dict):
                    flow = row['flow']
                    if 'Flow Duration' in features.columns:
                        features.at[idx, 'Flow Duration'] = flow.get('age', 0)
                    if 'Total Fwd Packets' in features.columns:
                        features.at[idx, 'Total Fwd Packets'] = flow.get('pkts_toserver', 0)
                    if 'Total Backward Packets' in features.columns:
                        features.at[idx, 'Total Backward Packets'] = flow.get('pkts_toclient', 0)
                    if 'Total Length of Fwd Packets' in features.columns:
                        features.at[idx, 'Total Length of Fwd Packets'] = flow.get('bytes_toserver', 0)
                    if 'Total Length of Bwd Packets' in features.columns:
                        features.at[idx, 'Total Length of Bwd Packets'] = flow.get('bytes_toclient', 0)
                    
                    # Calculate derived metrics
                    total_packets = flow.get('pkts_toserver', 0) + flow.get('pkts_toclient', 0)
                    total_bytes = flow.get('bytes_toserver', 0) + flow.get('bytes_toclient', 0)
                    
                    # Average Packet Size
                    if 'Average Packet Size' in features.columns:
                        features.at[idx, 'Average Packet Size'] = total_bytes / total_packets if total_packets > 0 else 0
                        
                    # Forward and backward packet size calculations
                    if flow.get('pkts_toserver', 0) > 0:
                        avg_pkt_size_fwd = flow['bytes_toserver'] / flow['pkts_toserver']
                    else:
                        avg_pkt_size_fwd = 0
                        
                    if 'Avg Fwd Segment Size' in features.columns:
                        features.at[idx, 'Avg Fwd Segment Size'] = avg_pkt_size_fwd
                        
                    if 'Fwd Packet Length Mean' in features.columns:
                        features.at[idx, 'Fwd Packet Length Mean'] = avg_pkt_size_fwd
                        
                    if 'Fwd Packet Length Max' in features.columns:
                        features.at[idx, 'Fwd Packet Length Max'] = avg_pkt_size_fwd * 1.5  # Estimation
                        
                    if 'Fwd Packet Length Min' in features.columns:
                        features.at[idx, 'Fwd Packet Length Min'] = avg_pkt_size_fwd * 0.5  # Estimation
                        
                    if 'Fwd Packet Length Std' in features.columns:
                        features.at[idx, 'Fwd Packet Length Std'] = avg_pkt_size_fwd * 0.2  # Estimation
                    
                    # Backward metrics
                    if flow.get('pkts_toclient', 0) > 0:
                        avg_pkt_size_bwd = flow['bytes_toclient'] / flow['pkts_toclient']
                    else:
                        avg_pkt_size_bwd = 0
                        
                    if 'Avg Bwd Segment Size' in features.columns:
                        features.at[idx, 'Avg Bwd Segment Size'] = avg_pkt_size_bwd
                        
                    if 'Bwd Packet Length Mean' in features.columns:
                        features.at[idx, 'Bwd Packet Length Mean'] = avg_pkt_size_bwd
                        
                    if 'Bwd Packet Length Max' in features.columns:
                        features.at[idx, 'Bwd Packet Length Max'] = avg_pkt_size_bwd * 1.5  # Estimation
                        
                    if 'Bwd Packet Length Min' in features.columns:
                        features.at[idx, 'Bwd Packet Length Min'] = avg_pkt_size_bwd * 0.5  # Estimation
                        
                    if 'Bwd Packet Length Std' in features.columns:
                        features.at[idx, 'Bwd Packet Length Std'] = avg_pkt_size_bwd * 0.2  # Estimation
                    
                    # Overall packet metrics
                    if total_packets > 0:
                        avg_pkt_size = total_bytes / total_packets
                    else:
                        avg_pkt_size = 0
                        
                    if 'Packet Length Mean' in features.columns:
                        features.at[idx, 'Packet Length Mean'] = avg_pkt_size
                        
                    if 'Packet Length Variance' in features.columns:
                        features.at[idx, 'Packet Length Variance'] = (avg_pkt_size_fwd - avg_pkt_size_bwd)**2 if avg_pkt_size_bwd > 0 else 0
                        
                    if 'Packet Length Std' in features.columns:
                        features.at[idx, 'Packet Length Std'] = np.sqrt(max(0, features.at[idx, 'Packet Length Variance'])) if 'Packet Length Variance' in features.columns else 0
                        
                    if 'Min Packet Length' in features.columns:
                        features.at[idx, 'Min Packet Length'] = min(avg_pkt_size_fwd * 0.5, avg_pkt_size_bwd * 0.5) if avg_pkt_size_bwd > 0 else avg_pkt_size_fwd * 0.5
                        
                    if 'Max Packet Length' in features.columns:
                        features.at[idx, 'Max Packet Length'] = max(avg_pkt_size_fwd * 1.5, avg_pkt_size_bwd * 1.5) if avg_pkt_size_bwd > 0 else avg_pkt_size_fwd * 1.5
                    
                    # Subflow features
                    if 'Subflow Fwd Packets' in features.columns:
                        features.at[idx, 'Subflow Fwd Packets'] = flow.get('pkts_toserver', 0)
                        
                    if 'Subflow Fwd Bytes' in features.columns:
                        features.at[idx, 'Subflow Fwd Bytes'] = flow.get('bytes_toserver', 0)
                        
                    if 'Subflow Bwd Packets' in features.columns:
                        features.at[idx, 'Subflow Bwd Packets'] = flow.get('pkts_toclient', 0)
                        
                    if 'Subflow Bwd Bytes' in features.columns:
                        features.at[idx, 'Subflow Bwd Bytes'] = flow.get('bytes_toclient', 0)
                    
                    # Header length estimation (typical TCP/IP header is ~40 bytes)
                    header_size = 40
                    if 'Fwd Header Length' in features.columns:
                        features.at[idx, 'Fwd Header Length'] = header_size * flow.get('pkts_toserver', 0)
                        
                    if 'Fwd Header Length.1' in features.columns:
                        features.at[idx, 'Fwd Header Length.1'] = features.at[idx, 'Fwd Header Length'] if 'Fwd Header Length' in features.columns else 0
                        
                    if 'Bwd Header Length' in features.columns:
                        features.at[idx, 'Bwd Header Length'] = header_size * flow.get('pkts_toclient', 0)
                    
                    # Flow rates
                    duration = flow.get('age', 0)
                    if duration > 0:
                        if 'Flow Bytes/s' in features.columns:
                            features.at[idx, 'Flow Bytes/s'] = total_bytes / duration
                            
                        if 'Flow Packets/s' in features.columns:
                            features.at[idx, 'Flow Packets/s'] = total_packets / duration
                            
                        if 'Fwd Packets/s' in features.columns:
                            features.at[idx, 'Fwd Packets/s'] = flow.get('pkts_toserver', 0) / duration
                            
                        if 'Bwd Packets/s' in features.columns:
                            features.at[idx, 'Bwd Packets/s'] = flow.get('pkts_toclient', 0) / duration
                    
                    # Flow IAT (Inter-Arrival Time) statistics
                    if duration > 0 and total_packets > 2:
                        avg_iat = duration / (total_packets - 1)
                        
                        if 'Flow IAT Mean' in features.columns:
                            features.at[idx, 'Flow IAT Mean'] = avg_iat
                            
                        if 'Flow IAT Std' in features.columns:
                            features.at[idx, 'Flow IAT Std'] = avg_iat * 0.3  # Estimation
                            
                        if 'Flow IAT Max' in features.columns:
                            features.at[idx, 'Flow IAT Max'] = avg_iat * 3  # Estimation
                            
                        if 'Flow IAT Min' in features.columns:
                            features.at[idx, 'Flow IAT Min'] = avg_iat * 0.1  # Estimation
                        
                        # Forward IAT
                        if flow.get('pkts_toserver', 0) > 1:
                            fwd_avg_iat = duration / (flow.get('pkts_toserver', 0) - 1)
                            
                            if 'Fwd IAT Total' in features.columns:
                                features.at[idx, 'Fwd IAT Total'] = duration
                                
                            if 'Fwd IAT Mean' in features.columns:
                                features.at[idx, 'Fwd IAT Mean'] = fwd_avg_iat
                                
                            if 'Fwd IAT Std' in features.columns:
                                features.at[idx, 'Fwd IAT Std'] = fwd_avg_iat * 0.3  # Estimation
                                
                            if 'Fwd IAT Max' in features.columns:
                                features.at[idx, 'Fwd IAT Max'] = fwd_avg_iat * 3  # Estimation
                                
                            if 'Fwd IAT Min' in features.columns:
                                features.at[idx, 'Fwd IAT Min'] = fwd_avg_iat * 0.1  # Estimation
                        
                        # Backward IAT
                        if flow.get('pkts_toclient', 0) > 1:
                            bwd_avg_iat = duration / (flow.get('pkts_toclient', 0) - 1)
                            
                            if 'Bwd IAT Total' in features.columns:
                                features.at[idx, 'Bwd IAT Total'] = duration
                                
                            if 'Bwd IAT Mean' in features.columns:
                                features.at[idx, 'Bwd IAT Mean'] = bwd_avg_iat
                                
                            if 'Bwd IAT Std' in features.columns:
                                features.at[idx, 'Bwd IAT Std'] = bwd_avg_iat * 0.3  # Estimation
                                
                            if 'Bwd IAT Max' in features.columns:
                                features.at[idx, 'Bwd IAT Max'] = bwd_avg_iat * 3  # Estimation
                                
                            if 'Bwd IAT Min' in features.columns:
                                features.at[idx, 'Bwd IAT Min'] = bwd_avg_iat * 0.1  # Estimation
                    
                    # Activity metrics
                    if 'Active Mean' in features.columns:
                        features.at[idx, 'Active Mean'] = duration * 0.7  # Estimation
                        
                    if 'Active Std' in features.columns:
                        features.at[idx, 'Active Std'] = duration * 0.2  # Estimation
                        
                    if 'Active Max' in features.columns:
                        features.at[idx, 'Active Max'] = duration  # Estimation
                        
                    if 'Active Min' in features.columns:
                        features.at[idx, 'Active Min'] = duration * 0.4  # Estimation
                        
                    if 'Idle Mean' in features.columns:
                        features.at[idx, 'Idle Mean'] = duration * 0.3  # Estimation
                        
                    if 'Idle Std' in features.columns:
                        features.at[idx, 'Idle Std'] = duration * 0.1  # Estimation
                        
                    if 'Idle Max' in features.columns:
                        features.at[idx, 'Idle Max'] = duration * 0.5  # Estimation
                        
                    if 'Idle Min' in features.columns:
                        features.at[idx, 'Idle Min'] = duration * 0.1  # Estimation
                    
                    # Additional metrics
                    if 'act_data_pkt_fwd' in features.columns:
                        features.at[idx, 'act_data_pkt_fwd'] = flow.get('pkts_toserver', 0) * 0.8  # Estimation
                        
                    if 'min_seg_size_forward' in features.columns:
                        features.at[idx, 'min_seg_size_forward'] = avg_pkt_size_fwd * 0.5 if avg_pkt_size_fwd > 0 else 40  # Estimation
                    
                    # Init window bytes
                    if 'Init_Win_bytes_forward' in features.columns:
                        features.at[idx, 'Init_Win_bytes_forward'] = 65535  # Default window size
                        
                    if 'Init_Win_bytes_backward' in features.columns:
                        features.at[idx, 'Init_Win_bytes_backward'] = 65535  # Default window size
                
                # Extract TCP information if available
                if 'tcp' in row and isinstance(row['tcp'], dict):
                    tcp = row['tcp']
                    # Extract TCP flags
                    if 'FIN Flag Count' in features.columns:
                        features.at[idx, 'FIN Flag Count'] = 1 if tcp.get('fin', False) else 0
                        
                    if 'SYN Flag Count' in features.columns:
                        features.at[idx, 'SYN Flag Count'] = 1 if tcp.get('syn', False) else 0
                        
                    if 'RST Flag Count' in features.columns:
                        features.at[idx, 'RST Flag Count'] = 1 if tcp.get('rst', False) else 0
                        
                    if 'PSH Flag Count' in features.columns:
                        features.at[idx, 'PSH Flag Count'] = 1 if tcp.get('psh', False) else 0
                        
                    if 'ACK Flag Count' in features.columns:
                        features.at[idx, 'ACK Flag Count'] = 1 if tcp.get('ack', False) else 0
                        
                    if 'URG Flag Count' in features.columns:
                        features.at[idx, 'URG Flag Count'] = 1 if tcp.get('urg', False) else 0
                        
                    if 'CWE Flag Count' in features.columns:
                        features.at[idx, 'CWE Flag Count'] = 1 if tcp.get('cwe', False) else 0
                        
                    if 'ECE Flag Count' in features.columns:
                        features.at[idx, 'ECE Flag Count'] = 1 if tcp.get('ece', False) else 0
                    
                    # Extract TCP window information
                    if ('tcp_options' in tcp and isinstance(tcp['tcp_options'], list) and 
                        'Init_Win_bytes_forward' in features.columns):
                        for option in tcp['tcp_options']:
                            if option.get('type') == 'window scale':
                                features.at[idx, 'Init_Win_bytes_forward'] = option.get('value', 65535)
                                break
                
                # Additional binary features that might be needed
                if 'SimillarHTTP' in features.columns:
                    features.at[idx, 'SimillarHTTP'] = 1 if row.get('app_proto') == 'http' else 0
                    
                if 'Inbound' in features.columns:
                    # Estimate if this is inbound based on destination port being a common server port
                    common_server_ports = [80, 443, 22, 21, 25, 53]
                    features.at[idx, 'Inbound'] = 1 if row.get('dest_port') in common_server_ports else 0
            
            # Reorder columns to match selected_features
            if self.selected_features:
                # Verify all required features are present
                missing_features = [f for f in self.selected_features if f not in features.columns]
                if missing_features:
                    print(f"Warning: Missing features for classification: {missing_features}")
                    # Add any missing features with default value of 0
                    for feature in missing_features:
                        features[feature] = 0
                        
                # Ensure column order matches the expected order
                return features[self.selected_features]
            
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
        print("\n" + "*"*40)
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
        # self.print_summary()
    
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
        
        self.print_summary()
    
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