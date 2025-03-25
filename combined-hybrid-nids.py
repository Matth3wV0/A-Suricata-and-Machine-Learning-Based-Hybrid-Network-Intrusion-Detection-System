#!/usr/bin/env python3
"""
Enhanced Hybrid Network Intrusion Detection System (NIDS)
Combining Suricata Signature-Based Detection with Machine Learning Anomaly Detection

This script combines the best features of hybrid_nids.py and test_hybrid_nids.py
to create a comprehensive NIDS that uses both signature-based detection from Suricata
and anomaly detection from machine learning.

Features:
- Data preprocessing and feature engineering
- Model training on CICIDS2017 dataset
- Advanced Suricata alert integration and analysis
- Comprehensive anomaly detection and visualization
- Detailed reporting with HTML reports and CSV exports
- ELK Stack integration preparation
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import joblib
import traceback
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_nids.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Hybrid-NIDS")

class HybridNIDS:
    def __init__(self, cicids_path=None, eve_json_path=None, model_path=None, binary_classification=True, include_all_alerts=True):
        """
        Initialize the Hybrid NIDS system.
        
        Args:
            cicids_path (str): Path to the CICIDS2017 dataset directory
            eve_json_path (str): Path to the Suricata eve.json file
            model_path (str): Path to save/load the trained model
            binary_classification (bool): Whether to use binary or multi-class classification
            include_all_alerts (bool): Whether to include all Suricata alerts in the output
        """
        self.cicids_path = cicids_path
        self.eve_json_path = eve_json_path
        self.model_path = model_path
        self.binary_classification = binary_classification
        self.include_all_alerts = include_all_alerts
        self.model = None
        self.scaler = None
        self.pca = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.attack_types = None
        self.features = None

    def load_cicids2017(self):
        """
        Load and preprocess the CICIDS2017 dataset.
        """
        logger.info("Loading CICIDS2017 dataset...")
        
        # Check if dataset path exists
        if not self.cicids_path or not os.path.exists(self.cicids_path):
            logger.error(f"CICIDS2017 dataset path not found: {self.cicids_path}")
            raise FileNotFoundError(f"CICIDS2017 dataset path not found: {self.cicids_path}")
        
        # Load CSV files from the directory
        csv_files = [f for f in os.listdir(self.cicids_path) if f.endswith('.csv')]
        if not csv_files:
            logger.error(f"No CSV files found in {self.cicids_path}")
            raise FileNotFoundError(f"No CSV files found in {self.cicids_path}")
        
        # Load and concatenate all CSV files
        dataframes = []
        for file in csv_files:
            file_path = os.path.join(self.cicids_path, file)
            logger.info(f"Loading {file_path}")
            try:
                df = pd.read_csv(file_path, low_memory=False)
                dataframes.append(df)
            except Exception as e:
                logger.warning(f"Error loading {file}: {str(e)}")
        
        if not dataframes:
            logger.error("Failed to load any CSV files")
            raise ValueError("Failed to load any CSV files")
        
        # Concatenate all dataframes
        data = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Loaded dataset with shape: {data.shape}")
        
        return data

    def preprocess_data(self, data):
        """
        Preprocess the CICIDS2017 dataset.
        
        Args:
            data (pd.DataFrame): The loaded CICIDS2017 dataset
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        logger.info("Preprocessing data...")
        
        # Clean column names
        data.columns = data.columns.str.strip()
        
        # Handle missing values
        logger.info("Handling missing values...")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Count missing values per column
        missing_values = data.isnull().sum()
        logger.info(f"Columns with missing values: {missing_values[missing_values > 0].to_dict()}")
        
        # Fill missing values with median (for numeric columns)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                median_value = data[col].median()
                data[col].fillna(median_value, inplace=True)
        
        # Remove duplicates
        logger.info("Removing duplicate records...")
        initial_count = len(data)
        data.drop_duplicates(inplace=True)
        logger.info(f"Removed {initial_count - len(data)} duplicate records")
        
        # Map target variable (Label column)
        logger.info("Mapping attack types...")
        if 'Label' in data.columns:
            attack_map = {
                'BENIGN': 'BENIGN',
                'DDoS': 'DDoS',
                'DoS Hulk': 'DoS',
                'DoS GoldenEye': 'DoS',
                'DoS slowloris': 'DoS',
                'DoS Slowhttptest': 'DoS',
                'PortScan': 'Port Scan',
                'FTP-Patator': 'Brute Force',
                'SSH-Patator': 'Brute Force',
                'Bot': 'Bot',
                'Web Attack Brute Force': 'Web Attack',
                'Web Attack XSS': 'Web Attack',
                'Web Attack Sql Injection': 'Web Attack',
                'Web Attack � Brute Force': 'Web Attack',
                'Web Attack � XSS': 'Web Attack',
                'Web Attack � Sql Injection': 'Web Attack',
                'Infiltration': 'Infiltration',
                'Heartbleed': 'Heartbleed'
            }
            data['Attack Type'] = data['Label'].map(lambda x: attack_map.get(x, x))
            data.drop('Label', axis=1, inplace=True)
        elif 'Attack Type' not in data.columns:
            logger.error("Neither 'Label' nor 'Attack Type' column found in the dataset")
            raise ValueError("Neither 'Label' nor 'Attack Type' column found in the dataset")
        
        # Create a binary column for normal vs attack
        data['is_attack'] = np.where(data['Attack Type'] == 'BENIGN', 0, 1)
        
        # Store unique attack types for later use
        self.attack_types = data['Attack Type'].unique()
        logger.info(f"Attack types in the dataset: {sorted([str(t) for t in self.attack_types])}")
        
        # Dropping columns with zero variance
        zero_var_cols = data.columns[data.nunique() == 1]
        data.drop(columns=zero_var_cols, inplace=True, errors='ignore')
        logger.info(f"Dropped {len(zero_var_cols)} columns with zero variance")
        
        # Memory optimization
        logger.info("Optimizing memory usage...")
        for col in data.select_dtypes(include=['float64']).columns:
            data[col] = data[col].astype('float32')
        for col in data.select_dtypes(include=['int64']).columns:
            data[col] = data[col].astype('int32')
        
        logger.info(f"Preprocessed data shape: {data.shape}")
        return data

    def prepare_training_data(self, data, binary_classification=None, balance_data=True):
        """
        Prepare training data for the machine learning model.
        
        Args:
            data (pd.DataFrame): Preprocessed data
            binary_classification (bool): If True, trains a binary classifier (normal vs attack)
            balance_data (bool): If True, balances the dataset
            
        Returns:
            X_train, X_test, y_train, y_test: Split datasets for training
        """
        logger.info("Preparing training data...")
        
        if binary_classification is None:
            binary_classification = self.binary_classification
        
        # Define target variable based on classification type
        if binary_classification:
            target = 'is_attack'
            logger.info("Using binary classification (normal vs attack)")
        else:
            target = 'Attack Type'
            logger.info(f"Using multi-class classification with {len(data[target].unique())} classes")
        
        # Select features (drop non-feature columns)
        non_features = ['Attack Type', 'is_attack']
        features = [col for col in data.columns if col not in non_features]
        self.features = features
        
        X = data[features]
        y = data[target]
        
        # Apply balancing if requested
        if balance_data:
            logger.info("Balancing dataset...")
            if binary_classification:
                # For binary classification, downsample the majority class
                normal_samples = data[data[target] == 0]
                attack_samples = data[data[target] == 1]
                
                # Balance by downsampling the majority class
                if len(normal_samples) > len(attack_samples):
                    normal_samples = normal_samples.sample(len(attack_samples), random_state=42)
                else:
                    attack_samples = attack_samples.sample(len(normal_samples), random_state=42)
                
                balanced_data = pd.concat([normal_samples, attack_samples])
                X = balanced_data[features]
                y = balanced_data[target]
            else:
                # For multi-class, sample a fixed number from each class
                balanced_data = []
                sample_size = 5000  # You can adjust this as needed
                
                for attack_type in data[target].unique():
                    class_data = data[data[target] == attack_type]
                    if len(class_data) > sample_size:
                        class_data = class_data.sample(sample_size, random_state=42)
                    balanced_data.append(class_data)
                
                balanced_data = pd.concat(balanced_data)
                X = balanced_data[features]
                y = balanced_data[target]
            
            logger.info(f"Balanced data shape: {X.shape}")
        
        # Feature scaling
        logger.info("Applying feature scaling...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        logger.info("Applying PCA for dimensionality reduction...")
        # Use half of the features as components, or adjust as needed
        n_components = min(len(features) // 2, 30)  
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        logger.info(f"PCA reduced dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]}")
        logger.info(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.2%}")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Store for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train=None, y_train=None):
        """
        Train the machine learning model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            trained model
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        
        if X_train is None or y_train is None:
            logger.error("Training data not available. Call prepare_training_data() first.")
            raise ValueError("Training data not available")
        
        logger.info("Training Random Forest classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train, y_train)
        self.model = model
        
        # Save the model if path is provided
        if self.model_path:
            self.save_model()
        
        return model

    def evaluate_model(self, X_test=None, y_test=None):
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        
        if X_test is None or y_test is None:
            logger.error("Test data not available. Call prepare_training_data() first.")
            raise ValueError("Test data not available")
        
        if self.model is None:
            logger.error("Model not trained. Call train_model() first.")
            raise ValueError("Model not trained")
        
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Create output directory for visualization
        vis_dir = "model_evaluation"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': [f"PC{i+1}" for i in range(len(self.model.feature_importances_))],
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
            plt.title('Top 20 Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'feature_importance.png'))
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix
        }

    def save_model(self):
        """
        Save the trained model and preprocessing components.
        """
        if self.model is None:
            logger.error("Model not trained. Call train_model() first.")
            raise ValueError("Model not trained")
        
        if not self.model_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model_path = f"hybrid_nids_model_{timestamp}"
        
        # Create directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Save model and preprocessing components
        joblib.dump(self.model, os.path.join(self.model_path, 'model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_path, 'scaler.pkl'))
        joblib.dump(self.pca, os.path.join(self.model_path, 'pca.pkl'))
        
        # Save feature names
        with open(os.path.join(self.model_path, 'features.json'), 'w') as f:
            json.dump(self.features, f)
        
        logger.info(f"Model and preprocessing components saved to {self.model_path}")

    def load_model(self):
        """
        Load a previously trained model and preprocessing components.
        """
        if not self.model_path or not os.path.exists(self.model_path):
            logger.error(f"Model path not found: {self.model_path}")
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        # Determine if this is a directory or direct model file
        if os.path.isdir(self.model_path):
            model_file = os.path.join(self.model_path, 'model.pkl')
            scaler_file = os.path.join(self.model_path, 'scaler.pkl')
            pca_file = os.path.join(self.model_path, 'pca.pkl')
            features_file = os.path.join(self.model_path, 'features.json')
        else:
            # Direct model file - use the same directory for other components
            model_file = self.model_path
            model_dir = os.path.dirname(self.model_path)
            scaler_file = os.path.join(model_dir, 'scaler.pkl')
            pca_file = os.path.join(model_dir, 'pca.pkl')
            features_file = os.path.join(model_dir, 'features.json')
        
        # Load model
        try:
            self.model = joblib.load(model_file)
            logger.info(f"Model loaded from {model_file}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Load scaler
        try:
            self.scaler = joblib.load(scaler_file)
            logger.info(f"Scaler loaded from {scaler_file}")
        except Exception as e:
            logger.warning(f"Error loading scaler: {str(e)}. A default scaler will be used.")
            self.scaler = StandardScaler()
        
        # Load PCA
        try:
            self.pca = joblib.load(pca_file)
            logger.info(f"PCA loaded from {pca_file}")
        except Exception as e:
            logger.warning(f"Error loading PCA: {str(e)}. Dimensionality reduction may not be applied correctly.")
            self.pca = None
        
        # Load feature names
        try:
            with open(features_file, 'r') as f:
                self.features = json.load(f)
            logger.info(f"Loaded {len(self.features)} feature names")
        except Exception as e:
            logger.warning(f"Error loading feature names: {str(e)}. Feature matching may be impaired.")
            self.features = None
        
        logger.info("Model and components loaded successfully")

    def parse_eve_json(self):
        """
        Parse the Suricata eve.json file and extract events.
        
        Returns:
            pd.DataFrame: Processed Suricata events
        """
        if not self.eve_json_path or not os.path.exists(self.eve_json_path):
            logger.error(f"Eve.json file not found: {self.eve_json_path}")
            raise FileNotFoundError(f"Eve.json file not found: {self.eve_json_path}")
        
        logger.info(f"Parsing Suricata eve.json: {self.eve_json_path}")
        
        # Read eve.json events
        events = []
        with open(self.eve_json_path, 'r') as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line: {line[:100]}...")
        
        logger.info(f"Parsed {len(events)} events from eve.json")
        
        # Process events to extract Suricata alert signatures and relevant fields
        for event in events:
            # Extract alert data and flatten it for easier access
            if 'alert' in event and isinstance(event['alert'], dict):
                for key, value in event['alert'].items():
                    event[f'alert.{key}'] = value
            
            # Extract flow data and flatten it for easier access
            if 'flow' in event and isinstance(event['flow'], dict):
                for key, value in event['flow'].items():
                    event[f'flow.{key}'] = value
            
            # Extract protocol specific fields for different event types
            for protocol in ['http', 'dns', 'tls', 'ssh', 'smtp', 'ftp']:
                if protocol in event and isinstance(event[protocol], dict):
                    for key, value in event[protocol].items():
                        event[f'{protocol}.{key}'] = value
        
        # Convert to DataFrame
        df = pd.json_normalize(events)
        
        if len(df) > 0:
            logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
            
            # Log event types distribution
            if 'event_type' in df.columns:
                event_type_counts = df['event_type'].value_counts()
                logger.info(f"Event types distribution:\n{event_type_counts}")
            
            # Log information about alert events
            alert_events = df[df['event_type'] == 'alert'] if 'event_type' in df.columns else pd.DataFrame()
            if not alert_events.empty:
                logger.info(f"Found {len(alert_events)} alert events")
                
                # Log alert categories if available
                if 'alert.category' in alert_events.columns:
                    alert_categories = alert_events['alert.category'].value_counts().head(10)
                    logger.info(f"Top 10 alert categories:\n{alert_categories}")
        else:
            logger.warning("No events found in eve.json")
        
        return df

    def extract_features_from_eve(self, eve_df):
        """
        Extract features from Suricata eve.json data that match CICIDS2017 features.
        
        Args:
            eve_df (pd.DataFrame): DataFrame of parsed eve.json events
            
        Returns:
            pd.DataFrame: Features extracted from eve.json
        """
        logger.info("Extracting features from Suricata eve.json")
        
        # Create an empty DataFrame to store extracted features
        extracted_features = pd.DataFrame()
        
        # Identify the event types
        event_types = eve_df['event_type'].unique()
        logger.info(f"Event types in eve.json: {event_types}")
        
        # Extract basic flow features (common to most events)
        
        # Timestamp
        if 'timestamp' in eve_df.columns:
            extracted_features['timestamp_str'] = eve_df['timestamp']
            
            # Store the original timestamp string - we'll calculate durations differently
            try:
                # Handle timestamp parsing more safely
                extracted_features['timestamp'] = pd.to_datetime(extracted_features['timestamp_str'], utc=True)
            except Exception as e:
                logger.warning(f"Error parsing timestamps: {str(e)}. Using string representation instead.")
                extracted_features['timestamp'] = extracted_features['timestamp_str']
        
        # Source and destination IP
        if 'src_ip' in eve_df.columns:
            extracted_features['src_ip'] = eve_df['src_ip']
        if 'dest_ip' in eve_df.columns:
            extracted_features['dest_ip'] = eve_df['dest_ip']
        
        # Source and destination port
        if 'src_port' in eve_df.columns:
            extracted_features['src_port'] = eve_df['src_port']
        if 'dest_port' in eve_df.columns:
            extracted_features['dest_port'] = eve_df['dest_port']
        
        # Protocol
        if 'proto' in eve_df.columns:
            extracted_features['proto'] = eve_df['proto']
            # One-hot encode protocol
            proto_dummies = pd.get_dummies(extracted_features['proto'], prefix='proto')
            extracted_features = pd.concat([extracted_features, proto_dummies], axis=1)
        
        # Extract alert information
        for alert_col in [col for col in eve_df.columns if 'alert' in col]:
            extracted_features[alert_col] = eve_df[alert_col]
        
        # Extract event_type information
        extracted_features['event_type'] = eve_df['event_type']
        
        # Flow ID (can be used to group packets in the same flow)
        if 'flow_id' in eve_df.columns:
            extracted_features['flow_id'] = eve_df['flow_id']
        
        # PCAP count (useful for packet-level analysis)
        if 'pcap_cnt' in eve_df.columns:
            extracted_features['pcap_cnt'] = eve_df['pcap_cnt']
        
        # Calculate flow duration
        if 'flow_id' in extracted_features.columns and 'timestamp' in extracted_features.columns:
            try:
                # Only try this if timestamp is a datetime type
                if pd.api.types.is_datetime64_any_dtype(extracted_features['timestamp']):
                    # Group by flow_id and calculate min and max timestamp
                    flow_durations = extracted_features.groupby('flow_id')['timestamp'].agg(['min', 'max'])
                    flow_durations['duration'] = (flow_durations['max'] - flow_durations['min']).dt.total_seconds()
                    
                    # Map duration back to original rows
                    flow_id_to_duration = flow_durations['duration'].to_dict()
                    extracted_features['flow_duration'] = extracted_features['flow_id'].map(flow_id_to_duration)
                else:
                    # Alternative approach for non-datetime timestamps
                    logger.warning("Timestamp is not a datetime type. Using flow packet counts as duration proxy.")
                    extracted_features['flow_duration'] = 0  # Default value
            except Exception as e:
                logger.warning(f"Error calculating flow durations: {str(e)}. Using default value.")
                extracted_features['flow_duration'] = 0  # Default value
        
        # Count packets per flow
        if 'flow_id' in extracted_features.columns:
            flow_packet_counts = extracted_features['flow_id'].value_counts().to_dict()
            extracted_features['flow_packets'] = extracted_features['flow_id'].map(flow_packet_counts)
        
        # Extract flow.* fields if available
        flow_fields = [col for col in eve_df.columns if col.startswith('flow.')]
        for field in flow_fields:
            if field in eve_df.columns:
                extracted_features[field] = eve_df[field]
        
        # Extract specific flow metrics
        if 'flow.bytes_toserver' in eve_df.columns:
            extracted_features['Total Length of Fwd Packets'] = eve_df['flow.bytes_toserver']
        if 'flow.bytes_toclient' in eve_df.columns:
            extracted_features['Total Length of Bwd Packets'] = eve_df['flow.bytes_toclient']
        if 'flow.pkts_toserver' in eve_df.columns:
            extracted_features['Total Fwd Packets'] = eve_df['flow.pkts_toserver']
        if 'flow.pkts_toclient' in eve_df.columns:
            extracted_features['Total Backward Packets'] = eve_df['flow.pkts_toclient']
        
        # Extract TLS/QUIC features if available
        tls_features = eve_df[eve_df['event_type'].isin(['tls', 'quic'])]
        if not tls_features.empty:
            # Create a map of flow_id to TLS features
            flow_to_tls = {}
            
            for _, row in tls_features.iterrows():
                flow_id = row.get('flow_id')
                if flow_id:
                    # Check if TLS or QUIC specific fields exist
                    tls_data = {}
                    
                    # TLS fields
                    if 'tls.version' in row:
                        tls_data['tls_version'] = row['tls.version']
                    elif 'tls' in row and isinstance(row['tls'], dict) and 'version' in row['tls']:
                        tls_data['tls_version'] = row['tls']['version']
                    
                    # SNI field
                    if 'tls.sni' in row:
                        tls_data['sni'] = row['tls.sni']
                    elif 'tls' in row and isinstance(row['tls'], dict) and 'sni' in row['tls']:
                        tls_data['sni'] = row['tls']['sni']
                    elif 'quic' in row and isinstance(row['quic'], dict) and 'sni' in row['quic']:
                        tls_data['sni'] = row['quic']['sni']
                    
                    # JA3 hash
                    if 'tls.ja3.hash' in row:
                        tls_data['ja3_hash'] = row['tls.ja3.hash']
                    elif 'tls' in row and isinstance(row['tls'], dict) and 'ja3' in row['tls'] and 'hash' in row['tls']['ja3']:
                        tls_data['ja3_hash'] = row['tls']['ja3']['hash']
                    elif 'quic' in row and isinstance(row['quic'], dict) and 'ja3' in row['quic'] and 'hash' in row['quic']['ja3']:
                        tls_data['ja3_hash'] = row['quic']['ja3']['hash']
                    
                    # Add data to the map
                    flow_to_tls[flow_id] = tls_data
            
            # Now map these features back to the extracted_features DataFrame
            if 'flow_id' in extracted_features.columns:
                # Create a has_tls_or_quic column
                extracted_features['has_tls_or_quic'] = extracted_features['flow_id'].isin(flow_to_tls.keys()).astype(int)
                
                # Function to get TLS data for a flow
                def get_tls_data(flow_id, field):
                    tls_data = flow_to_tls.get(flow_id, {})
                    return tls_data.get(field, np.nan)
                
                # Map TLS features
                extracted_features['tls_version'] = extracted_features['flow_id'].apply(
                    lambda x: get_tls_data(x, 'tls_version'))
                extracted_features['sni'] = extracted_features['flow_id'].apply(
                    lambda x: get_tls_data(x, 'sni'))
                extracted_features['ja3_hash'] = extracted_features['flow_id'].apply(
                    lambda x: get_tls_data(x, 'ja3_hash'))
        
        # Extract DNS features
        dns_features = eve_df[eve_df['event_type'] == 'dns']
        if not dns_features.empty:
            # Create a map of flow_id to DNS features
            flow_to_dns = {}
            
            for _, row in dns_features.iterrows():
                flow_id = row.get('flow_id')
                if flow_id:
                    # Check if DNS specific fields exist
                    dns_data = {}
                    
                    # DNS query type
                    if 'dns.type' in row:
                        dns_data['dns_type'] = row['dns.type']
                    elif 'dns' in row and isinstance(row['dns'], dict) and 'type' in row['dns']:
                        dns_data['dns_type'] = row['dns']['type']
                    
                    # DNS rrname (domain)
                    if 'dns.rrname' in row:
                        dns_data['dns_rrname'] = row['dns.rrname']
                    elif 'dns' in row and isinstance(row['dns'], dict) and 'rrname' in row['dns']:
                        dns_data['dns_rrname'] = row['dns']['rrname']
                    
                    # DNS rrtype
                    if 'dns.rrtype' in row:
                        dns_data['dns_rrtype'] = row['dns.rrtype']
                    elif 'dns' in row and isinstance(row['dns'], dict) and 'rrtype' in row['dns']:
                        dns_data['dns_rrtype'] = row['dns']['rrtype']
                    
                    # Add data to the map
                    flow_to_dns[flow_id] = dns_data
            
            # Now map these features back to the extracted_features DataFrame
            if 'flow_id' in extracted_features.columns:
                # Create a has_dns column
                extracted_features['has_dns'] = extracted_features['flow_id'].isin(flow_to_dns.keys()).astype(int)
                
                # Function to get DNS data for a flow
                def get_dns_data(flow_id, field):
                    dns_data = flow_to_dns.get(flow_id, {})
                    return dns_data.get(field, np.nan)
                
                # Map DNS features
                extracted_features['dns_type'] = extracted_features['flow_id'].apply(
                    lambda x: get_dns_data(x, 'dns_type'))
                extracted_features['dns_rrname'] = extracted_features['flow_id'].apply(
                    lambda x: get_dns_data(x, 'dns_rrname'))
                extracted_features['dns_rrtype'] = extracted_features['flow_id'].apply(
                    lambda x: get_dns_data(x, 'dns_rrtype'))
        
        # Extract HTTP features
        http_features = eve_df[eve_df['event_type'] == 'http']
        if not http_features.empty:
            # Create a map of flow_id to HTTP features
            flow_to_http = {}
            
            for _, row in http_features.iterrows():
                flow_id = row.get('flow_id')
                if flow_id:
                    # Check if HTTP specific fields exist
                    http_data = {}
                    
                    # HTTP method
                    if 'http.http_method' in row:
                        http_data['http_method'] = row['http.http_method']
                    elif 'http' in row and isinstance(row['http'], dict) and 'http_method' in row['http']:
                        http_data['http_method'] = row['http']['http_method']
                    
                    # HTTP status
                    if 'http.status' in row:
                        http_data['http_status'] = row['http.status']
                    elif 'http' in row and isinstance(row['http'], dict) and 'status' in row['http']:
                        http_data['http_status'] = row['http']['status']
                    
                    # HTTP content type
                    if 'http.http_content_type' in row:
                        http_data['http_content_type'] = row['http.http_content_type']
                    elif 'http' in row and isinstance(row['http'], dict) and 'http_content_type' in row['http']:
                        http_data['http_content_type'] = row['http']['http_content_type']
                    
                    # HTTP length
                    if 'http.length' in row:
                        http_data['http_length'] = row['http.length']
                    elif 'http' in row and isinstance(row['http'], dict) and 'length' in row['http']:
                        http_data['http_length'] = row['http']['length']
                    
                    # Add data to the map
                    flow_to_http[flow_id] = http_data
            
            # Now map these features back to the extracted_features DataFrame
            if 'flow_id' in extracted_features.columns:
                # Create a has_http column
                extracted_features['has_http'] = extracted_features['flow_id'].isin(flow_to_http.keys()).astype(int)
                
                # Function to get HTTP data for a flow
                def get_http_data(flow_id, field):
                    http_data = flow_to_http.get(flow_id, {})
                    return http_data.get(field, np.nan)
                
                # Map HTTP features
                extracted_features['http_method'] = extracted_features['flow_id'].apply(
                    lambda x: get_http_data(x, 'http_method'))
                extracted_features['http_status'] = extracted_features['flow_id'].apply(
                    lambda x: get_http_data(x, 'http_status'))
                extracted_features['http_content_type'] = extracted_features['flow_id'].apply(
                    lambda x: get_http_data(x, 'http_content_type'))
                extracted_features['http_length'] = extracted_features['flow_id'].apply(
                    lambda x: get_http_data(x, 'http_length'))
        
        # Extract categorical columns for one-hot encoding
        categorical_cols = ['dns_type', 'dns_rrtype', 'tls_version', 'http_method', 'http_content_type']
        # Fix for categorical columns one-hot encoding
        for col in categorical_cols:
            if col in extracted_features.columns:
                # Fill NaN values with 'UNKNOWN'
                extracted_features[col] = extracted_features[col].fillna('UNKNOWN')
                
                # One-hot encode
                try:
                    dummies = pd.get_dummies(extracted_features[col], prefix=col)
                    extracted_features = pd.concat([extracted_features, dummies], axis=1)
                except Exception as e:
                    logger.warning(f"Error one-hot encoding column {col}: {str(e)}")
        
        # Drop columns that cannot be used for ML
        cols_to_drop = ['timestamp', 'timestamp_str', 'src_ip', 'dest_ip', 'dns_rrname', 'sni', 'ja3_hash', 'flow_id', 'proto']
        cols_to_drop = [col for col in cols_to_drop if col in extracted_features.columns]
        
        # Don't drop alert-related columns
        cols_to_drop = [col for col in cols_to_drop if not col.startswith('alert')]
        
        # Create a separate DataFrame for non-ML columns (like alerts)
        alert_cols = [col for col in extracted_features.columns if col.startswith('alert')]
        extracted_features_ml = extracted_features.drop(columns=cols_to_drop, errors='ignore')
        
        # Handle missing values for ML features
        extracted_features_ml.fillna(0, inplace=True)
        
        logger.info(f"Extracted features shape: {extracted_features_ml.shape}")
        logger.info(f"Features: {extracted_features_ml.columns.tolist()}")
        
        return extracted_features_ml
    
    def predict_suricata_events(self):
        """
        Apply the trained model to predict anomalies in Suricata events.
        
        Returns:
            pd.DataFrame: Suricata events with predictions and attack classifications
        """
        if self.model is None:
            logger.error("Model not trained. Call train_model() or load_model() first.")
            raise ValueError("Model not trained")
        
        logger.info("Predicting anomalies in Suricata events...")
        
        # Parse eve.json
        eve_df = self.parse_eve_json()
        
        # Extract features
        features_df = self.extract_features_from_eve(eve_df)
        
        # Keep a copy of the original features
        eve_with_features = eve_df.copy()
        
        # Add extracted features to the original DataFrame for reference
        for col in features_df.columns:
            if col in eve_with_features.columns:
                # Avoid column name collision by adding a prefix
                eve_with_features[f'feature_{col}'] = features_df[col].values
            else:
                eve_with_features[col] = features_df[col].values
        
        # Verify we have at least some data
        if features_df.empty:
            logger.error("No features extracted from Suricata events")
            raise ValueError("No features extracted from Suricata events")
        
        try:
            # Get exact feature names from the model
            if hasattr(self.model, 'feature_names_in_'):
                # Use the exact feature names from the model
                model_features = self.model.feature_names_in_
                logger.info(f"Using model's exact feature names: {len(model_features)} features")
            else:
                # If model doesn't have feature_names_in_, check if features.json exists
                # This is a fallback for older scikit-learn versions
                if self.features is not None:
                    model_features = self.features
                    logger.info(f"Using features from features.json: {len(model_features)} features")
                else:
                    # Last resort: use a default set of CICIDS features
                    logger.warning("No feature names found. Using default CICIDS features")
                    model_features = [
                        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
                        'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
                        'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
                        'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
                        'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
                        'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
                        'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags'
                    ]
            
            # Log a few feature names for debugging
            if len(model_features) > 5:
                logger.info(f"First 5 model features: {', '.join(str(f) for f in model_features[:5])}")
            
            # Create empty DataFrame with exact feature names from model
            model_ready_features = pd.DataFrame(0.0, index=np.arange(len(features_df)), 
                                            columns=model_features)
            
            # Create a mapping of extracted features to model features
            # These mappings should be adjusted based on your actual feature names
            feature_mapping = {}
            
            # Basic features that might match directly
            if 'src_port' in features_df.columns:
                if 'Source Port' in model_features:
                    feature_mapping['src_port'] = 'Source Port'
                # Check for alternate names
                elif 'Src Port' in model_features:
                    feature_mapping['src_port'] = 'Src Port'
            
            if 'dest_port' in features_df.columns:
                if 'Destination Port' in model_features:
                    feature_mapping['dest_port'] = 'Destination Port'
                # Check for alternate names
                elif 'Dst Port' in model_features:
                    feature_mapping['dest_port'] = 'Dst Port'
            
            if 'flow_duration' in features_df.columns:
                if 'Flow Duration' in model_features:
                    feature_mapping['flow_duration'] = 'Flow Duration'
            
            if 'flow_packets' in features_df.columns:
                if 'Total Fwd Packets' in model_features:
                    feature_mapping['flow_packets'] = 'Total Fwd Packets'
                elif 'Total Packets' in model_features:
                    feature_mapping['flow_packets'] = 'Total Packets'
            
            # Direct mappings for standard CICIDS2017 features
            for cicids_feature in ['Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                                'Total Fwd Packets', 'Total Backward Packets']:
                if cicids_feature in features_df.columns and cicids_feature in model_features:
                    feature_mapping[cicids_feature] = cicids_feature
            
            # Apply the direct mappings
            for suricata_feature, model_feature in feature_mapping.items():
                if suricata_feature in features_df.columns and model_feature in model_ready_features.columns:
                    model_ready_features[model_feature] = features_df[suricata_feature].values
            
            # Protocol mappings (using .loc to avoid Series truth value error)
            protocol_feature = None
            for candidate in ['Protocol', 'protocol', 'IP Protocol']:
                if candidate in model_features:
                    protocol_feature = candidate
                    break
            
            if protocol_feature is not None:
                if 'proto_TCP' in features_df.columns:
                    model_ready_features.loc[features_df['proto_TCP'] == 1, protocol_feature] = 6  # TCP
                if 'proto_UDP' in features_df.columns:
                    model_ready_features.loc[features_df['proto_UDP'] == 1, protocol_feature] = 17  # UDP
                if 'proto_ICMP' in features_df.columns:
                    model_ready_features.loc[features_df['proto_ICMP'] == 1, protocol_feature] = 1  # ICMP
            
            # Calculate flow-related features if available in the model
            flow_features = {
                'Flow Packets/s': lambda df: df['flow_packets'] / df['flow_duration'].replace(0, 1),
                'Flow IAT Mean': lambda df: df['flow_duration'] / df['flow_packets'].replace(0, 2),
                'Flow IAT Std': lambda df: df['flow_duration'] / (df['flow_packets'].replace(0, 2) * 2),
                'Fwd Packets/s': lambda df: df.get('Total Fwd Packets', df['flow_packets']/2) / df['flow_duration'].replace(0, 1),
                'Bwd Packets/s': lambda df: df.get('Total Backward Packets', df['flow_packets']/2) / df['flow_duration'].replace(0, 1),
                # Add more calculated features as needed
            }
            
            for model_feature, calc_func in flow_features.items():
                if model_feature in model_features and 'flow_packets' in features_df.columns and 'flow_duration' in features_df.columns:
                    try:
                        model_ready_features[model_feature] = calc_func(features_df)
                    except Exception as e:
                        logger.warning(f"Error calculating {model_feature}: {str(e)}")
            
            # Apply feature scaling
            logger.info("Applying feature scaling...")
            X_scaled = self.scaler.transform(model_ready_features)
            
            # Apply PCA transformation if available
            if self.pca is not None:
                logger.info("Applying PCA transformation...")
                X_pca = self.pca.transform(X_scaled)
            else:
                logger.warning("PCA not available. Using scaled features directly.")
                X_pca = X_scaled
            
            # Make predictions
            logger.info("Making predictions...")
            predictions = self.model.predict(X_pca)
            probabilities = self.model.predict_proba(X_pca)
            
            # Add predictions to the DataFrame
            eve_with_features['prediction'] = predictions
            
            # Add probability columns
            for i, class_label in enumerate(self.model.classes_):
                eve_with_features[f'prob_{class_label}'] = probabilities[:, i]
            
            # Add a confidence column (max probability)
            eve_with_features['confidence'] = probabilities.max(axis=1)
            
            # Define attack type mapping based on binary/multi-class model
            if self.binary_classification:
                # Binary classification
                attack_mapping = {
                    0: "BENIGN",
                    1: "ATTACK"
                }
            else:
                # Multi-class classification
                # If your model was trained on CICIDS2017, these labels should match those classes
                # Update this mapping according to your model's output classes
                attack_mapping = {
                    0: "BENIGN",
                    1: "Bot",
                    2: "Brute Force",
                    3: "DDoS",
                    4: "DoS",
                    5: "Heartbleed",
                    6: "Infiltration",
                    7: "Port Scan",
                    8: "Web Attack"
                }
            
            # If we have the actual attack_types mapping from training, use that instead
            if hasattr(self, 'attack_types') and self.attack_types is not None:
                # Create mapping from numeric predictions to attack type names
                unique_attack_types = sorted(list(set(self.attack_types)))
                attack_mapping = {i: attack_type for i, attack_type in enumerate(unique_attack_types)}
                logger.info(f"Using attack type mapping: {attack_mapping}")
            
            # Add attack type column (map numeric prediction to attack name)
            eve_with_features['attack_type'] = eve_with_features['prediction'].map(
                lambda x: attack_mapping.get(x, "Unknown")
            )
            
            # Identify potential anomalies based on confidence
            threshold = 0.8  # Configurable threshold
            eve_with_features['is_anomaly'] = (eve_with_features['confidence'] < threshold) | (eve_with_features['attack_type'] != "BENIGN")
            
            # Create a more meaningful anomaly description
            eve_with_features['anomaly_description'] = 'Normal'
            
            # For alerts (anomalies with known attack type)
            known_attack_mask = (eve_with_features['is_anomaly'] == True) & (eve_with_features['attack_type'] != "BENIGN")
            if not known_attack_mask.empty:
                eve_with_features.loc[known_attack_mask, 'anomaly_description'] = eve_with_features.loc[known_attack_mask, 'attack_type']
            
            # For uncertain/unknown anomalies (low confidence but not classified as an attack)
            uncertain_mask = (eve_with_features['is_anomaly'] == True) & (eve_with_features['attack_type'] == "BENIGN")
            if not uncertain_mask.empty:
                eve_with_features.loc[uncertain_mask, 'anomaly_description'] = "Unknown Anomaly"
            
            # Include Suricata alerts in anomaly description
            # This integrates better with the original test_hybrid_nids.py functionality
            if 'alert.signature' in eve_with_features.columns:
                # Find rows with Suricata alerts
                signature_mask = ~eve_with_features['alert.signature'].isna()
                
                # We can apply a different logic based on inclusion parameter
                if self.include_all_alerts:
                    # Consider all Suricata alerts as anomalies
                    eve_with_features.loc[signature_mask, 'is_anomaly'] = True
                    eve_with_features.loc[signature_mask, 'anomaly_description'] = "Suricata: " + eve_with_features.loc[signature_mask, 'alert.signature'].astype(str)
                else:
                    # Only add Suricata info to already detected anomalies
                    anomaly_with_signature = signature_mask & eve_with_features['is_anomaly']
                    if not anomaly_with_signature.empty:
                        eve_with_features.loc[anomaly_with_signature, 'anomaly_description'] = "Suricata: " + eve_with_features.loc[anomaly_with_signature, 'alert.signature'].astype(str)
            
            # Count anomalies by type for reporting
            anomaly_counts = eve_with_features[eve_with_features['is_anomaly'] == True]['anomaly_description'].value_counts()
            
            logger.info(f"Predicted {len(eve_with_features)} events")
            logger.info(f"Detected {eve_with_features['is_anomaly'].sum()} potential anomalies")
            if not anomaly_counts.empty:
                logger.info(f"Anomalies by type:\n{anomaly_counts}")
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(traceback.format_exc())
            # Create minimal prediction columns to allow the rest of the process to continue
            eve_with_features['prediction'] = 0
            eve_with_features['confidence'] = 1.0
            eve_with_features['is_anomaly'] = False
            eve_with_features['attack_type'] = "UNKNOWN"
            eve_with_features['anomaly_description'] = "Error in prediction"
            if len(self.model.classes_) > 0:
                for i in range(len(self.model.classes_)):
                    eve_with_features[f'prob_{i}'] = 0.0
                eve_with_features[f'prob_0'] = 1.0  # Default to class 0
            
            logger.warning("Using default values for predictions due to error")
        
        return eve_with_features
        
    def analyze_anomalies(self, predictions_df):
        """
        Analyze detected anomalies in more detail with attack type classification.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions
            
        Returns:
            pd.DataFrame: Anomaly analysis results
        """
        if 'is_anomaly' not in predictions_df.columns:
            logger.error("Predictions DataFrame does not contain 'is_anomaly' column")
            raise ValueError("Predictions DataFrame does not contain required columns")
        
        logger.info("Analyzing detected anomalies...")
        
        # Filter anomalies
        anomalies = predictions_df[predictions_df['is_anomaly'] == True]
        
        if anomalies.empty:
            logger.info("No anomalies detected")
            return pd.DataFrame()
        
        logger.info(f"Analyzing {len(anomalies)} anomalies")
        
        # Group anomalies by attack type/description
        if 'anomaly_description' in anomalies.columns:
            attack_type_counts = anomalies['anomaly_description'].value_counts()
            logger.info(f"Anomalies by type:\n{attack_type_counts}")
        elif 'attack_type' in anomalies.columns:
            attack_type_counts = anomalies['attack_type'].value_counts()
            logger.info(f"Anomalies by attack type:\n{attack_type_counts}")
        
        # Group anomalies by event type
        if 'event_type' in anomalies.columns:
            event_type_counts = anomalies['event_type'].value_counts()
            logger.info(f"Anomalies by event type:\n{event_type_counts}")
        
        # Group anomalies by source IP
        if 'src_ip' in anomalies.columns:
            src_ip_counts = anomalies['src_ip'].value_counts().head(10)
            logger.info(f"Top 10 source IPs with anomalies:\n{src_ip_counts}")
        
        # Group anomalies by destination IP
        if 'dest_ip' in anomalies.columns:
            dest_ip_counts = anomalies['dest_ip'].value_counts().head(10)
            logger.info(f"Top 10 destination IPs with anomalies:\n{dest_ip_counts}")
        
        # Group anomalies by protocol
        if 'proto' in anomalies.columns:
            proto_counts = anomalies['proto'].value_counts()
            logger.info(f"Anomalies by protocol:\n{proto_counts}")
        
        # Create correlation matrix of attack types and protocols
        if 'anomaly_description' in anomalies.columns and 'proto' in anomalies.columns:
            try:
                attack_proto_matrix = pd.crosstab(anomalies['anomaly_description'], anomalies['proto'])
                logger.info(f"Attack type vs Protocol distribution:\n{attack_proto_matrix}")
            except Exception as e:
                logger.warning(f"Error creating attack-protocol matrix: {str(e)}")
        
        # Create a time series of anomalies if timestamp is available
        if 'timestamp' in anomalies.columns:
            try:
                # Check if timestamp is a datetime type
                if pd.api.types.is_datetime64_any_dtype(anomalies['timestamp']):
                    # Create a copy for time series analysis
                    time_anomalies = anomalies.copy()
                    time_anomalies.set_index('timestamp', inplace=True)
                    
                    # Create output directory for visualizations
                    vis_dir = "anomaly_analysis"
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # Resample by hour - catch potential errors
                    try:
                        hourly_counts = time_anomalies.resample('H').size()
                        
                        # Plot time series
                        plt.figure(figsize=(12, 6))
                        hourly_counts.plot()
                        plt.title('Anomalies Over Time')
                        plt.xlabel('Time')
                        plt.ylabel('Number of Anomalies')
                        plt.grid(True)
                        plt.savefig(os.path.join(vis_dir, 'anomalies_time_series.png'))
                        logger.info(f"Created time series plot: {os.path.join(vis_dir, 'anomalies_time_series.png')}")
                        
                        # If we have attack types, create a stacked time series
                        if 'anomaly_description' in time_anomalies.columns:
                            try:
                                # Create pivot table for attack types over time
                                attack_time_pivot = time_anomalies.pivot_table(
                                    index=time_anomalies.index.floor('H'),
                                    columns='anomaly_description', 
                                    aggfunc='size', 
                                    fill_value=0
                                )
                                
                                # Plot stacked time series
                                plt.figure(figsize=(14, 8))
                                attack_time_pivot.plot.area(stacked=True, alpha=0.7)
                                plt.title('Attack Types Over Time')
                                plt.xlabel('Time')
                                plt.ylabel('Number of Anomalies')
                                plt.grid(True)
                                plt.legend(title='Attack Type', loc='center left', bbox_to_anchor=(1, 0.5))
                                plt.tight_layout()
                                plt.savefig(os.path.join(vis_dir, 'attack_types_time_series.png'))
                                logger.info(f"Created attack types time series plot: {os.path.join(vis_dir, 'attack_types_time_series.png')}")
                            except Exception as e:
                                logger.warning(f"Error creating attack types time series: {str(e)}")                                
                                logger.info(f"Created attack types time series plot: {os.path.join(vis_dir, 'attack_types_time_series.png')}")
                            except Exception as e:
                                logger.warning(f"Error creating attack types time series: {str(e)}")
                        
                    except Exception as e:
                        logger.warning(f"Error creating time series: {str(e)}")
                else:
                    logger.warning("Timestamp is not a datetime type. Skipping time series analysis.")
            except Exception as e:
                logger.warning(f"Error processing timestamp data: {str(e)}")
        
        # Create a detailed anomaly report
        anomaly_report = {}
        
        if 'anomaly_description' in anomalies.columns:
            anomaly_report['attack_types'] = anomalies['anomaly_description'].value_counts().to_dict()
        elif 'attack_type' in anomalies.columns:
            anomaly_report['attack_types'] = anomalies['attack_type'].value_counts().to_dict()
        
        if 'event_type' in anomalies.columns:
            anomaly_report['event_types'] = anomalies['event_type'].value_counts().to_dict()
        
        if 'src_ip' in anomalies.columns:
            anomaly_report['source_ip_count'] = len(anomalies['src_ip'].unique())
            anomaly_report['top_source_ips'] = anomalies['src_ip'].value_counts().head(5).to_dict()
        
        if 'dest_ip' in anomalies.columns:
            anomaly_report['dest_ip_count'] = len(anomalies['dest_ip'].unique())
            anomaly_report['top_dest_ips'] = anomalies['dest_ip'].value_counts().head(5).to_dict()
        
        # Calculate average confidence for each attack type
        if 'confidence' in anomalies.columns and 'anomaly_description' in anomalies.columns:
            confidence_by_type = anomalies.groupby('anomaly_description')['confidence'].mean().to_dict()
            anomaly_report['confidence_by_type'] = confidence_by_type
        
        # Count Suricata alerts vs ML-only anomalies
        if 'anomaly_description' in anomalies.columns:
            suricata_alerts = anomalies[anomalies['anomaly_description'].str.contains('Suricata:', na=False)]
            ml_anomalies = anomalies[~anomalies['anomaly_description'].str.contains('Suricata:', na=False)]
            
            anomaly_report['suricata_alert_count'] = len(suricata_alerts)
            anomaly_report['ml_anomaly_count'] = len(ml_anomalies)
            
            logger.info(f"Suricata alerts: {len(suricata_alerts)}")
            logger.info(f"ML-only anomalies: {len(ml_anomalies)}")
        
        # Convert to DataFrame for easier handling
        anomaly_report_df = pd.DataFrame([anomaly_report])
        
        return anomalies

    def _generate_html_report(self, anomalies, all_events):
        """
        Generate an HTML report of attack classifications
        
        Args:
            anomalies: DataFrame with anomaly events
            all_events: DataFrame with all events
        
        Returns:
            str: HTML report content
        """
        if anomalies.empty:
            return "<html><body><h1>No anomalies detected</h1></body></html>"
        
        # Start HTML document
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hybrid NIDS Attack Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #2c3e50; }
                .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .chart { margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; padding: 10px; }
                .alert { background-color: #ffdddd; }
                .high { background-color: #ff6666; color: white; }
                .medium { background-color: #ffb366; }
                .low { background-color: #ffff99; }
            </style>
        </head>
        <body>
            <h1>Hybrid NIDS Attack Classification Report</h1>
            <div class="summary">
        """
        
        # Add summary section
        total_events = len(all_events)
        total_anomalies = len(anomalies)
        anomaly_percentage = (total_anomalies / total_events) * 100 if total_events > 0 else 0
        
        html += f"""
                <h2>Summary</h2>
                <p><strong>Total Events Analyzed:</strong> {total_events}</p>
                <p><strong>Anomalies Detected:</strong> {total_anomalies} ({anomaly_percentage:.2f}%)</p>
                <p><strong>Analysis Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add attack type breakdown
        if 'attack_type' in anomalies.columns:
            html += """
            <h2>Attack Type Distribution</h2>
            <table>
                <tr>
                    <th>Attack Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            """
            
            attack_counts = anomalies['attack_type'].value_counts()
            for attack, count in attack_counts.items():
                percentage = (count / total_anomalies) * 100
                html += f"""
                <tr>
                    <td>{attack}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add anomaly description breakdown
        if 'anomaly_description' in anomalies.columns:
            html += """
            <h2>Anomaly Description Distribution</h2>
            <table>
                <tr>
                    <th>Description</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            """
            
            desc_counts = anomalies['anomaly_description'].value_counts()
            for desc, count in desc_counts.items():
                percentage = (count / total_anomalies) * 100
                html += f"""
                <tr>
                    <td>{desc}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add top source IPs
        if 'src_ip' in anomalies.columns:
            html += """
            <h2>Top Source IPs</h2>
            <table>
                <tr>
                    <th>Source IP</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Top Attack Type</th>
                </tr>
            """
            
            src_ip_counts = anomalies['src_ip'].value_counts().head(10)
            for src_ip, count in src_ip_counts.items():
                percentage = (count / total_anomalies) * 100
                
                # Find the top attack type for this source IP
                top_attack = "Unknown"
                if 'attack_type' in anomalies.columns:
                    ip_attacks = anomalies[anomalies['src_ip'] == src_ip]['attack_type'].value_counts()
                    if not ip_attacks.empty:
                        top_attack = ip_attacks.index[0]
                elif 'anomaly_description' in anomalies.columns:
                    ip_attacks = anomalies[anomalies['src_ip'] == src_ip]['anomaly_description'].value_counts()
                    if not ip_attacks.empty:
                        top_attack = ip_attacks.index[0]
                
                html += f"""
                <tr>
                    <td>{src_ip}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                    <td>{top_attack}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add top destination IPs
        if 'dest_ip' in anomalies.columns:
            html += """
            <h2>Top Destination IPs</h2>
            <table>
                <tr>
                    <th>Destination IP</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Top Attack Type</th>
                </tr>
            """
            
            dest_ip_counts = anomalies['dest_ip'].value_counts().head(10)
            for dest_ip, count in dest_ip_counts.items():
                percentage = (count / total_anomalies) * 100
                
                # Find the top attack type for this destination IP
                top_attack = "Unknown"
                if 'attack_type' in anomalies.columns:
                    ip_attacks = anomalies[anomalies['dest_ip'] == dest_ip]['attack_type'].value_counts()
                    if not ip_attacks.empty:
                        top_attack = ip_attacks.index[0]
                elif 'anomaly_description' in anomalies.columns:
                    ip_attacks = anomalies[anomalies['dest_ip'] == dest_ip]['anomaly_description'].value_counts()
                    if not ip_attacks.empty:
                        top_attack = ip_attacks.index[0]
                
                html += f"""
                <tr>
                    <td>{dest_ip}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                    <td>{top_attack}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add protocol distribution
        if 'proto' in anomalies.columns:
            html += """
            <h2>Protocol Distribution</h2>
            <table>
                <tr>
                    <th>Protocol</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            """
            
            proto_counts = anomalies['proto'].value_counts()
            for proto, count in proto_counts.items():
                percentage = (count / total_anomalies) * 100
                html += f"""
                <tr>
                    <td>{proto}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add event type distribution
        if 'event_type' in anomalies.columns:
            html += """
            <h2>Event Type Distribution</h2>
            <table>
                <tr>
                    <th>Event Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            """
            
            event_counts = anomalies['event_type'].value_counts()
            for event, count in event_counts.items():
                percentage = (count / total_anomalies) * 100
                html += f"""
                <tr>
                    <td>{event}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add top anomalies with details
        html += """
        <h2>Top Anomaly Events</h2>
        <table>
            <tr>
                <th>Time</th>
                <th>Source</th>
                <th>Destination</th>
                <th>Protocol</th>
                <th>Event Type</th>
                <th>Attack Type</th>
                <th>Confidence</th>
            </tr>
        """
        
        # Sort anomalies by confidence (lower confidence = more anomalous)
        sorted_anomalies = anomalies
        if 'confidence' in anomalies.columns:
            sorted_anomalies = anomalies.sort_values(by='confidence')
        
        # Show top 50 anomalies
        for idx, row in sorted_anomalies.head(50).iterrows():
            time_str = row.get('timestamp', 'N/A')
            src = f"{row.get('src_ip', 'N/A')}:{row.get('src_port', 'N/A')}" if 'src_ip' in row else 'N/A'
            dst = f"{row.get('dest_ip', 'N/A')}:{row.get('dest_port', 'N/A')}" if 'dest_ip' in row else 'N/A'
            proto = row.get('proto', 'N/A')
            event_type = row.get('event_type', 'N/A')
            attack_type = row.get('anomaly_description', row.get('attack_type', 'Unknown'))
            confidence = row.get('confidence', 'N/A')
            
            # Set row class based on confidence
            row_class = ""
            if isinstance(confidence, (int, float)):
                if confidence < 0.5:
                    row_class = "high"
                elif confidence < 0.7:
                    row_class = "medium"
                elif confidence < 0.9:
                    row_class = "low"
            
            html += f"""
            <tr class="{row_class}">
                <td>{time_str}</td>
                <td>{src}</td>
                <td>{dst}</td>
                <td>{proto}</td>
                <td>{event_type}</td>
                <td>{attack_type}</td>
                <td>{confidence}</td>
            </tr>
            """
        
        html += """
        </table>
        
        <h2>Recommendations</h2>
        <p>Based on the detected anomalies, consider the following actions:</p>
        <ul>
        """
        
        # Add attack-specific recommendations
        attack_types = set()
        if 'attack_type' in anomalies.columns:
            attack_types = set(anomalies['attack_type'])
        elif 'anomaly_description' in anomalies.columns:
            attack_types = set(anomalies['anomaly_description'])
        
        # Standard recommendations
        recommendations = []
        
        # General recommendation
        recommendations.append("Review Suricata rules and threshold settings to reduce false positives")
        
        # Attack-specific recommendations
        if "DDoS" in attack_types or "DoS" in attack_types or any("DoS" in str(at) for at in attack_types):
            recommendations.append("Implement rate limiting and traffic filtering for DoS/DDoS protection")
        
        if "Port Scan" in attack_types or any("scan" in str(at).lower() for at in attack_types):
            recommendations.append("Review firewall rules to limit exposure of sensitive ports")
        
        if "Brute Force" in attack_types or any("brute" in str(at).lower() for at in attack_types):
            recommendations.append("Implement account lockout policies and strong password requirements")
        
        if "Web Attack" in attack_types or any("web" in str(at).lower() for at in attack_types) or any("sql" in str(at).lower() for at in attack_types):
            recommendations.append("Apply web application firewall (WAF) rules and patch web applications")
        
        if "Bot" in attack_types or any("bot" in str(at).lower() for at in attack_types):
            recommendations.append("Check for infected hosts and implement network segmentation")
        
        if "Unknown Anomaly" in attack_types or any("unknown" in str(at).lower() for at in attack_types):
            recommendations.append("Investigate unknown anomalies with packet capture analysis")
        
        # Suricata-specific recommendations
        if any("suricata" in str(at).lower() for at in attack_types):
            recommendations.append("Review and update Suricata signatures for the detected threats")
        
        # Add recommendations to HTML
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        
        # Close HTML document
        html += """
        </ul>
        
        <div class="footer">
            <p>Generated by Hybrid NIDS - Combining Suricata and Machine Learning</p>
            <p>Report Time: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        
        </body>
        </html>
        """
        
        return html
    
    # def export_results(self, predictions_df, output_path=None):
    #     """
    #     Export analysis results to various formats.
        
    #     Args:
    #         predictions_df (pd.DataFrame): DataFrame with predictions
    #         output_path (str): Output directory
        
    #     Returns:
    #         str: Path to the exported results directory
    #     """
    #     if output_path is None:
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         output_path = f"hybrid_nids_results_{timestamp}"
        
    #     # Create directory if it doesn't exist
    #     os.makedirs(output_path, exist_ok=True)
        
    #     logger.info(f"Exporting results to {output_path}")
        
    #     # Create output subdirectories
    #     csv_dir = os.path.join(output_path, "csv")
    #     reports_dir = os.path.join(output_path, "reports")
    #     vis_dir = os.path.join(output_path, "visualizations")
        
    #     os.makedirs(csv_dir, exist_ok=True)
    #     os.makedirs(reports_dir, exist_ok=True)
    #     os.makedirs(vis_dir, exist_ok=True)
        
    #     # Copy visualization files if they exist
    #     for vis_file in ['anomalies_time_series.png', 'attack_types_time_series.png']:
    #         src_path = os.path.join("anomaly_analysis", vis_file)
    #         if os.path.exists(src_path):
    #             shutil.copy2(src_path, os.path.join(vis_dir, vis_file))
        
    #     # Export full results to CSV with proper escaping for special characters
    #     try:
    #         # Clean the DataFrame to ensure it can be written to CSV safely
    #         # Convert any complex objects to strings to avoid serialization issues
    #         safe_df = predictions_df.copy()
    #         for col in safe_df.columns:
    #             if safe_df[col].dtype == 'object':
    #                 # Convert complex objects to string representation
    #                 safe_df[col] = safe_df[col].apply(lambda x: str(x) if isinstance(x, (dict, list, tuple)) else x)
            
    #         # Use escapechar and quoting options to handle special characters
    #         safe_df.to_csv(
    #             os.path.join(csv_dir, 'predictions.csv'),
    #             index=False,
    #             escapechar='\\',
    #             quoting=1  # csv.QUOTE_ALL
    #         )
            
    #         # Export anomalies to a separate CSV
    #         if 'is_anomaly' in safe_df.columns:
    #             anomalies = safe_df[(safe_df['is_anomaly'] == True) & (safe_df['attack_type'] != 'BENIGN')]
    #             if not anomalies.empty:
    #                 anomalies.to_csv(
    #                     os.path.join(csv_dir, 'anomalies.csv'),
    #                     index=False,
    #                     escapechar='\\',
    #                     quoting=1  # csv.QUOTE_ALL
    #                 )
                
    #                 # Generate HTML report for better visualization
    #                 html_report = self._generate_html_report(anomalies, predictions_df)
    #                 with open(os.path.join(reports_dir, 'attack_report.html'), 'w') as f:
    #                     f.write(html_report)
                    
    #                 # Also export alerts separately
    #                 if any(col.startswith('alert') for col in safe_df.columns):
    #                     alerts = safe_df[safe_df['event_type'] == 'alert']
    #                     if not alerts.empty:
    #                         alerts.to_csv(
    #                             os.path.join(csv_dir, 'suricata_alerts.csv'),
    #                             index=False,
    #                             escapechar='\\',
    #                             quoting=1  # csv.QUOTE_ALL
    #                         )
                
    #                 # Export network flows separately
    #                 if 'event_type' in safe_df.columns:
    #                     flows = safe_df[safe_df['event_type'] == 'flow']
    #                     if not flows.empty:
    #                         flows.to_csv(
    #                             os.path.join(csv_dir, 'network_flows.csv'),
    #                             index=False,
    #                             escapechar='\\',
    #                             quoting=1  # csv.QUOTE_ALL
    #                         )
            
    #     except Exception as e:
    #         logger.error(f"Error exporting CSVs: {str(e)}")
    #         # Try a more basic export format
    #         try:
    #             # Export key columns only
    #             minimal_cols = ['event_type', 'is_anomaly', 'confidence', 'attack_type',
    #                             'src_ip', 'dest_ip', 'src_port', 'dest_port', 'proto']
    #             minimal_cols = [col for col in minimal_cols if col in predictions_df.columns]
                
    #             minimal_df = predictions_df[minimal_cols].copy()
    #             minimal_df.to_csv(
    #                 os.path.join(csv_dir, 'minimal_predictions.csv'),
    #                 index=False
    #             )
    #             logger.info("Exported minimal predictions CSV as fallback")
    #         except Exception as e2:
    #             logger.error(f"Error exporting minimal CSV: {str(e2)}")
    #             # Last resort - write a text summary
    #             with open(os.path.join(output_path, 'summary.txt'), 'w') as f:
    #                 f.write(f"Total events: {len(predictions_df)}\n")
    #                 if 'is_anomaly' in predictions_df.columns:
    #                     f.write(f"Potential anomalies: {predictions_df['is_anomaly'].sum()}\n")
    #                 f.write(f"Export time: {datetime.now().isoformat()}\n")
        
    #     # Export summary statistics
    #     try:
    #         summary = {
    #             'total_events': int(len(predictions_df)),
    #             'anomalies': int(predictions_df['is_anomaly'].sum()) if 'is_anomaly' in predictions_df.columns else 0,
    #             'timestamp': datetime.now().isoformat()
    #         }
            
    #         # Add predictions by class if available
    #         if 'prediction' in predictions_df.columns:
    #             predictions_counts = predictions_df['prediction'].value_counts().to_dict()
    #             for class_label, count in predictions_counts.items():
    #                 # Convert numpy types to Python native types for JSON serialization
    #                 if isinstance(class_label, (np.integer, np.floating)):
    #                     class_label = int(class_label) if isinstance(class_label, np.integer) else float(class_label)
    #                 summary[f'class_{class_label}'] = int(count)
            
    #         # Add Suricata alert statistics
    #         if 'event_type' in predictions_df.columns:
    #             alert_count = len(predictions_df[predictions_df['event_type'] == 'alert'])
    #             summary['suricata_alerts'] = alert_count
            
    #         # Add attack type statistics
    #         if 'attack_type' in predictions_df.columns:
    #             attack_counts = predictions_df[predictions_df['is_anomaly'] == True]['attack_type'].value_counts().to_dict()
    #             summary['attack_types'] = {str(k): int(v) for k, v in attack_counts.items()}
            
    #         # Export summary as JSON with custom serializer for numpy types
    #         with open(os.path.join(output_path, 'summary.json'), 'w') as f:
    #             json.dump(summary, f, indent=2, default=lambda x: 
    #                     int(x) if isinstance(x, np.integer) else
    #                     float(x) if isinstance(x, np.floating) else
    #                     x.tolist() if isinstance(x, np.ndarray) else
    #                     str(x))
    #     except Exception as e:
    #         logger.error(f"Error exporting summary JSON: {str(e)}")
        
    #     logger.info(f"Results exported to {output_path}")
        
    #     return output_path
    
    def export_results(self, predictions_df, output_path=None):
        """
        Export minimal results only when anomalies are detected.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions
            output_path (str): Output directory
        
        Returns:
            str: Path to the log file or None if no anomalies detected
        """
        # Check if there are any anomalies to report
        if 'is_anomaly' not in predictions_df.columns or not predictions_df['is_anomaly'].any():
            logger.info("No anomalies detected, skipping export")
            return None
        
        # Filter out BENIGN predictions to only capture actual attacks
        anomalies = predictions_df[(predictions_df['is_anomaly'] == True) & 
                                (predictions_df['attack_type'] != 'BENIGN')]
        
        if anomalies.empty:
            logger.info("No actual attacks detected, skipping export")
            return None
        
        # Log the detected anomalies
        logger.info(f"Exporting {len(anomalies)} detected anomalies")
        return self.log_detected_anomalies(anomalies)

    def log_detected_anomalies(self, anomalies_df):
        """
        Log detected anomalies to file without generating HTML or CSV reports.
        
        Args:
            anomalies_df (pd.DataFrame): DataFrame containing detected anomalies
        """
        if anomalies_df.empty:
            return
        
        # Create a minimal log directory if it doesn't exist
        log_dir = "anomaly_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"anomaly_log_{timestamp}.txt")
        
        # Write minimal information to the log file
        with open(log_file, 'w') as f:
            f.write(f"Anomalies detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total anomalies: {len(anomalies_df)}\n\n")
            
            # Write summary by attack type
            f.write("=== Attack Types ===\n")
            attack_counts = anomalies_df['attack_type'].value_counts()
            for attack, count in attack_counts.items():
                f.write(f"{attack}: {count}\n")
            
            # Write source IPs and their attack types
            f.write("\n=== Source IPs ===\n")
            for src_ip in anomalies_df['src_ip'].unique():
                ip_attacks = anomalies_df[anomalies_df['src_ip'] == src_ip]
                attack_types = ip_attacks['attack_type'].unique()
                f.write(f"{src_ip}: {len(ip_attacks)} events, attack types: {', '.join(map(str, attack_types))}\n")
            
            # Write detailed anomaly records
            f.write("\n=== Detailed Anomalies ===\n")
            for idx, row in anomalies_df.iterrows():
                f.write(f"Flow ID: {row.get('flow_id', 'N/A')}\n")
                f.write(f"  Source: {row.get('src_ip', 'N/A')}:{row.get('src_port', 'N/A')}\n")
                f.write(f"  Destination: {row.get('dest_ip', 'N/A')}:{row.get('dest_port', 'N/A')}\n")
                f.write(f"  Protocol: {row.get('proto', 'N/A')}\n")
                f.write(f"  Event Type: {row.get('event_type', 'N/A')}\n")
                f.write(f"  Attack Type: {row.get('attack_type', 'Unknown')}\n")
                f.write(f"  Confidence: {row.get('confidence', 'N/A')}\n")
                f.write("---\n")
        
        logger.info(f"Anomaly log saved to {log_file}")

    def analyze_real_time(self, interval=10, duration=None, blacklist_threshold=0.9):
        """
        Run continuous real-time analysis on Suricata events.
        
        Args:
            interval (int): Seconds between each analysis run
            duration (int): Total duration in seconds to run (None for indefinite)
            blacklist_threshold (float): Confidence threshold for blacklisting
            
        Returns:
            None
        """
        import time
        from datetime import datetime
        import subprocess
        import ipaddress
        
        logger.info(f"Starting real-time analysis with {interval}s interval")
        logger.info(f"Blacklisting IPs with confidence > {blacklist_threshold}")
        
        start_time = time.time()
        run_count = 0
        detected_anomalies = set()  # Track already detected anomalies
        blacklisted_ips = set()  # Track already blacklisted IPs
        
        try:
            while True:
                run_count += 1
                current_time = time.time()
                elapsed = current_time - start_time
                
                if duration is not None and elapsed > duration:
                    logger.info(f"Reached specified duration of {duration}s. Stopping.")
                    break
                
                logger.info(f"Run #{run_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                try:
                    # Make sure the model is loaded
                    if self.model is None:
                        self.load_model()
                    
                    # Analyze current Suricata events
                    predictions = self.predict_suricata_events()
                    
                    # Check for anomalies
                    if 'is_anomaly' in predictions.columns:
                        # Filter for new anomalies with attack_type not BENIGN
                        anomalies = predictions[(predictions['is_anomaly'] == True) & 
                                            (predictions['attack_type'] != 'BENIGN')]
                        
                        if not anomalies.empty:
                            # Check for high confidence attacks to blacklist
                            high_confidence_attacks = anomalies[anomalies['confidence'] < blacklist_threshold]
                            
                            # Extract unique flow IDs for detected anomalies
                            new_anomalies = set(anomalies['flow_id']) - detected_anomalies
                            
                            if new_anomalies:
                                logger.info(f"Detected {len(new_anomalies)} new anomalous flows")
                                detected_anomalies.update(new_anomalies)
                                
                                # Export only new anomalies to save resources
                                new_anomalies_df = anomalies[anomalies['flow_id'].isin(new_anomalies)]
                                
                                # Instead of full export, just log essential information
                                self.log_detected_anomalies(new_anomalies_df)
                                
                                # Process high confidence attacks for blacklisting
                                if not high_confidence_attacks.empty:
                                    for _, attack in high_confidence_attacks.iterrows():
                                        src_ip = attack.get('src_ip')
                                        attack_type = attack.get('attack_type', 'Unknown')
                                        confidence = attack.get('confidence', 1.0)
                                        
                                        # Skip if IP is private, loopback, etc.
                                        if src_ip and src_ip not in blacklisted_ips:
                                            try:
                                                ip = ipaddress.ip_address(src_ip)
                                                if not (ip.is_private or ip.is_loopback or ip.is_multicast):
                                                    logger.info(f"Blacklisting {src_ip} for {attack_type} attack (confidence: {confidence:.3f})")
                                                    
                                                    # Add to Suricata blacklist
                                                    self.add_ip_to_blacklist(src_ip)
                                                    blacklisted_ips.add(src_ip)
                                            except ValueError:
                                                logger.warning(f"Invalid IP address format: {src_ip}")
                            else:
                                logger.info("No new anomalies detected in this run")
                        else:
                            logger.info("No anomalies detected in this run")
                    
                except Exception as e:
                    logger.error(f"Error during analysis run #{run_count}: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # Wait for the next interval
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Real-time analysis stopped by user")
        
        logger.info(f"Real-time analysis complete. Detected {len(detected_anomalies)} anomalous flows.")
        logger.info(f"Blacklisted {len(blacklisted_ips)} IP addresses.")

def main():
    """
    Main function to run the Hybrid NIDS.
    """
    parser = argparse.ArgumentParser(description='Hybrid Network Intrusion Detection System')
    parser.add_argument('--cicids_path', help='Path to CICIDS2017 dataset directory')
    parser.add_argument('--eve_json', help='Path to Suricata eve.json file')
    parser.add_argument('--model_path', help='Path to save/load model')
    parser.add_argument('--output_path', help='Path to save results')
    parser.add_argument('--multiclass', action='store_true', help='Use multi-class classification')
    parser.add_argument('--include_all_alerts', action='store_true', help='Include all Suricata alerts in ML analysis')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--analyze', action='store_true', help='Analyze Suricata logs')
    parser.add_argument('--realtime', action='store_true', help='Run in real-time monitoring mode')
    parser.add_argument('--interval', type=int, default=10, help='Interval between analyses in seconds when in real-time mode')
    parser.add_argument('--duration', type=int, help='Duration to run real-time analysis in seconds')
    parser.add_argument('--threshold', type=float, default=0.9, help='Confidence threshold for blacklisting IPs')
    
    args = parser.parse_args()
    
    # Initialize the Hybrid NIDS
    nids = HybridNIDS(
        cicids_path=args.cicids_path,
        eve_json_path=args.eve_json,
        model_path=args.model_path,
        binary_classification=not args.multiclass,
        include_all_alerts=args.include_all_alerts
    )
    
    # Train or load the model
    if args.train:
        if not args.cicids_path:
            logger.error("--cicids_path is required for training")
            return
        
        try:
            # Load and preprocess data
            data = nids.load_cicids2017()
            processed_data = nids.preprocess_data(data)
            
            # Prepare training data
            X_train, X_test, y_train, y_test = nids.prepare_training_data(
                processed_data, 
                binary_classification=not args.multiclass
            )
            
            # Train model
            nids.train_model(X_train, y_train)
            
            # Evaluate model
            nids.evaluate_model(X_test, y_test)
            
            logger.info("Model training and evaluation completed")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    elif args.analyze or args.realtime:
        # Load the model
        if not args.model_path:
            logger.error("--model_path is required for analysis without training")
            return
        
        try:
            nids.load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    # Run real-time analysis mode
    if args.realtime:
        if not args.eve_json:
            logger.error("--eve_json is required for real-time analysis")
            return
        
        try:
            logger.info(f"Starting real-time analysis with {args.interval}s interval")
            nids.analyze_real_time(
                interval=args.interval, 
                duration=args.duration,
                blacklist_threshold=args.threshold
            )
        except Exception as e:
            logger.error(f"Error during real-time analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    # Analyze Suricata logs (one-time)
    elif args.analyze:
        if not args.eve_json:
            logger.error("--eve_json is required for analysis")
            return
        
        try:
            # Predict anomalies
            predictions = nids.predict_suricata_events()
            
            # Analyze anomalies
            anomalies = nids.analyze_anomalies(predictions)
            
            # Export results only if anomalies were detected (simplified output)
            output_path = nids.export_results(predictions, args.output_path)
            
            # Provide basic statistics
            if 'is_anomaly' in predictions.columns:
                total_events = len(predictions)
                # Only count actual attacks, not BENIGN traffic
                actual_attacks = predictions[(predictions['is_anomaly'] == True) & 
                                          (predictions['attack_type'] != 'BENIGN')]
                attack_count = len(actual_attacks)
                attack_percent = (attack_count / total_events) * 100
                
                logger.info(f"Analysis statistics:")
                logger.info(f"- Total events: {total_events}")
                logger.info(f"- Attacks detected: {attack_count} ({attack_percent:.2f}%)")
                
                # Count Suricata alerts vs ML anomalies
                if 'anomaly_description' in predictions.columns:
                    suricata_alerts = predictions[predictions['anomaly_description'].str.contains('Suricata:', na=False)]
                    ml_anomalies = predictions[(predictions['is_anomaly'] == True) & 
                                            (~predictions['anomaly_description'].str.contains('Suricata:', na=False)) &
                                            (predictions['attack_type'] != 'BENIGN')]
                    
                    logger.info(f"- Suricata alerts: {len(suricata_alerts)}")
                    logger.info(f"- ML-only attacks: {len(ml_anomalies)}")
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise

if __name__ == "__main__":
    main()

