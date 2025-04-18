#!/usr/bin/env python3
"""
Enhanced Hybrid NIDS with Session and Behavioral Awareness

This script combines signature-based detection (Suricata) with session-aware and 
behavioral anomaly detection to provide comprehensive network intrusion detection.

Usage:
    python hybrid_nids.py [--train <dataset_path>] 
                          [--analyze <path_to_suricata_json>] 
                          [--realtime <path_to_suricata_json>]
                          [--model_dir <model_directory>]
                          [--output <output_file>]
                          [--telegram]
"""

import argparse
import os
import sys
import json
import asyncio
import time
import datetime
import pickle
import logging
import pandas as pd
import numpy as np
from dataclasses import asdict
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from utils.dataset_balancer import DatasetBalancer, integrate_binary_balancing, integrate_multiclass_balancing
from telethon import TelegramClient
# Import custom modules
from suricata.suricata_parser import SuricataParser
from utils.adaptive_flow_features import AdaptiveFlowFeatureExtractor 
from utils.anomaly_detector import AnomalyDetector
from utils.telegram_alert import TelegramAlerter
from utils.service_whitelist import ServiceWhitelist
# Import new modules
from utils.session_manager import SessionManager, SuricataSession
from utils.behavioral_analyzer import BehavioralAnalyzer
from utils.flow_finalizer import FlowFinalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hybrid_nids.log')
    ]
)
logger = logging.getLogger('hybrid-nids')

# Load environment variables from .env file
load_dotenv()

# List of features to use based on matching table
ALIGNED_FEATURES = [
    "dest_port",          # Destination Port
    "duration",           # Flow Duration
    "total_fwd_packets",  # Total Fwd Packets
    "total_bwd_packets",  # Total Backward Packets
    "total_fwd_bytes",    # Total Length of Fwd Packets
    "total_bwd_bytes",    # Total Length of Bwd Packets
    "flow_bytes_per_sec", # Flow Bytes/s
    "flow_packets_per_sec", # Flow Packets/s
    "down_up_ratio"       # Down/Up Ratio
]

class HybridNIDS:
    """
    Enhanced Hybrid Network Intrusion Detection System with session and behavioral awareness
    that combines signature-based detection with advanced anomaly detection capabilities.
    """
    
    def __init__(self, model_dir='./model', telegram_enabled=False):
        """Initialize the Enhanced Hybrid NIDS."""
        self.model_dir = model_dir
        self.models = None
        self.telegram_enabled = telegram_enabled
        
        # Initialize TelegramAlerter if enabled
        # It will connect automatically in the background
        self.alerter = None
        if telegram_enabled:
            try:
                self.alerter = TelegramAlerter()
                logger.info("Telegram alerter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram alerter: {e}")
                self.alerter = None
        
        self.parser = SuricataParser()
        self.feature_extractor = AdaptiveFlowFeatureExtractor(ALIGNED_FEATURES)
        self.service_whitelist = ServiceWhitelist()
        
        # Get current directory for session file
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize new components with optimized parameters for SSH brute force detection
        self.session_manager = SessionManager(
            session_timeout=60,        # 1 minute timeout for regular sessions
            ssh_session_timeout=15,    # 15 seconds timeout for SSH sessions
            max_sessions=50000         # Maximum sessions to keep in memory
        )
        
        self.behavioral_analyzer = BehavioralAnalyzer(
            window_size=300,           # 5 minutes window for behavioral analysis
            cleanup_interval=30,       # Cleanup every 30 seconds - more frequent
            max_tracked_ips=10000      # Maximum IPs to track
        )
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to load models if they exist
        try:
            self.load_models()
            logger.info("Models loaded successfully.")
            
            # Initialize anomaly detector
            self.anomaly_detector = AnomalyDetector(model_dir=model_dir)
            
            # Initialize flow finalizer with optimized parameters
            self.flow_finalizer = FlowFinalizer(
                feature_extractor=self.feature_extractor,
                anomaly_detector=self.anomaly_detector,
                alert_callback=self.handle_alert,
                min_session_duration=0.0,      # Don't filter by duration
                zero_byte_threshold=2,         # Lower threshold (2 instead of 3)
                save_results=True,
                results_file="flow_results.csv"
            )
            
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            self.models = None
        
    def load_models(self):
        """Load trained models and related files."""
        logger.info(f"Loading models from {self.model_dir}...")
        
        result = {}
        
        # Load Decision Tree model
        with open(os.path.join(self.model_dir, 'dt_model.pkl'), 'rb') as f:
            result['dt_model'] = pickle.load(f)
        
        # Load Random Forest model
        with open(os.path.join(self.model_dir, 'rf_model.pkl'), 'rb') as f:
            result['rf_model'] = pickle.load(f)
            
        # Try to load XGBoost model if available
        try:
            with open(os.path.join(self.model_dir, 'xgb_model.pkl'), 'rb') as f:
                result['xgb_model'] = pickle.load(f)
            logger.info("XGBoost model loaded successfully.")
            
            # Load label encoder for XGBoost
            with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
                result['label_encoder'] = pickle.load(f)
        except FileNotFoundError:
            logger.warning("XGBoost model not found. Will use DT and RF only.")
        
        # Load scaler
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
            result['scaler'] = pickle.load(f)
        
        # Load baseline statistics
        with open(os.path.join(self.model_dir, 'baseline.json'), 'r') as f:
            result['baseline'] = json.load(f)
        
        self.models = result
        
        return result
    
    def train(self, dataset_path):
        """Train the machine learning models using CICIDS2017 dataset with balanced data."""
        logger.info(f"Training models using dataset: {dataset_path}")
        
        # Load and preprocess dataset
        df = self._load_and_preprocess_dataset(dataset_path)
        
        # First verify the original distribution
        if 'Label' in df.columns:
            orig_labels = df['Label']
            self._verify_labels(orig_labels, is_balanced=False)
            logger.info(f"Original dataset shape: {df.shape}")
        
        # Apply dataset balancing for binary classification
        logger.info("Applying dataset balancing for binary classification...")
        balanced_df = integrate_binary_balancing(df, target_col='Label', benign_value=0)
        logger.info(f"Balanced dataset shape: {balanced_df.shape}")
        
        # Prepare features and target from balanced dataset
        X, y = self._prepare_features_target(balanced_df)
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Verify the balanced data distribution
        self._verify_labels(y_train, is_balanced=True)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame for feature names
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Create label encoder for XGBoost
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # Train Decision Tree
        dt_model = self._train_decision_tree(X_train_scaled_df, y_train, X_test_scaled_df, y_test)
        
        # Train Random Forest
        rf_model = self._train_random_forest(X_train_scaled_df, y_train, X_test_scaled_df, y_test)
        
        # Train XGBoost
        xgb_model = self._train_xgboost(X_train_scaled_df, y_train_encoded, X_test_scaled_df, y_test_encoded, le)
        
        # Create baseline statistics
        baseline = self._create_baseline(X_train_scaled_df[y_train == 0])
        
        # Save models and related files
        self._save_models(dt_model, rf_model, xgb_model, scaler, le, baseline)
        
        # Load models
        self.load_models()
        
        logger.info("Training completed successfully.")
    
    def _verify_labels(self, y_train, is_balanced=True):
        """Verify label distribution and check for potential issues.
        
        Args:
            y_train: Training labels
            is_balanced: Flag indicating if the dataset has been balanced
        """
        unique, counts = np.unique(y_train, return_counts=True)
        label_counts = dict(zip(unique, counts))
        logger.info(f"LABEL DISTRIBUTION: {label_counts}")
        
        # Calculate the ratio between classes
        if 0 in label_counts and 1 in label_counts:
            ratio = label_counts[0] / label_counts[1]
            
            # For unbalanced datasets (original data), benign should be more common
            if not is_balanced and ratio < 1:
                logger.warning("WARNING: Found more attack samples than benign samples. Labels might be reversed!")
            
            # For balanced datasets, we expect roughly equal distribution
            if is_balanced:
                if ratio > 1.2 or ratio < 0.8:
                    logger.warning(f"WARNING: Imbalance detected in supposedly balanced data. Class ratio: {ratio:.2f}")
                else:
                    logger.info(f"Class distribution looks good for balanced data. Class ratio: {ratio:.2f}")
    
    def _load_and_preprocess_dataset(self, dataset_path):
        """
        Load and preprocess the CICIDS2017 dataset with enhanced cleaning.
        """
        logger.info("Loading and preprocessing dataset...")
        
        # Load dataset
        if os.path.isdir(dataset_path):
            # Load from directory (multiple CSV files)
            df = pd.DataFrame()
            for dirname, _, filenames in os.walk(dataset_path):
                for filename in filenames:
                    if filename.endswith('.csv'):
                        file_path = os.path.join(dirname, filename)
                        logger.info(f"Reading file: {file_path}")
                        df = pd.concat([df, pd.read_csv(file_path)], ignore_index=True)
        else:
            # Load from single CSV file
            df = pd.read_csv(dataset_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Log missing values
        if df.isna().sum().sum() > 0:
            logger.info(f"Missing values before imputation: {df.isna().sum().sum()}")
            
            # For numeric columns, fill with median
            numeric_cols = df.select_dtypes(include=['float', 'int']).columns
            for col in numeric_cols:
                if df[col].isna().sum() > 0:
                    logger.info(f"Column {col} with NaN or infinite values.")
                    # df[col].fillna(df[col].median(), inplace=True)
        
        # Drop rows that still have NaN values
        df.dropna(inplace=True)
        
        # Drop duplicates
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            logger.info(f"Removing {n_duplicates} duplicate rows")
            df.drop_duplicates(inplace=True)
        
        # Convert labels to binary (benign=0, attack=1) if 'Label' exists
        if 'Label' in df.columns:
            df['Label'] = np.where((df['Label'] == 'BENIGN') | (df['Label'] == 'benign'), 0, 1)
        elif 'label' in df.columns:
            df['Label'] = np.where((df['label'] == 'BENIGN') | (df['label'] == 'benign'), 0, 1)
            df.drop('label', axis=1, inplace=True)
        
        logger.info(f"Dataset loaded and preprocessed. Shape: {df.shape}")
        
        return df
    
    def _prepare_features_target(self, df):
        """Prepare features and target for model training."""
        logger.info("Preparing features and target...")
        
        # Map CICIDS2017 column names to Suricata feature names
        feature_mapping = {
            'Destination Port': 'dest_port',
            'Flow Duration': 'duration',
            'Total Fwd Packets': 'total_fwd_packets',
            'Total Backward Packets': 'total_bwd_packets',
            'Total Length of Fwd Packets': 'total_fwd_bytes',
            'Total Length of Bwd Packets': 'total_bwd_bytes',
            'Flow Bytes/s': 'flow_bytes_per_sec',
            'Flow Packets/s': 'flow_packets_per_sec',
            'Down/Up Ratio': 'down_up_ratio'
        }
        
        # Extract target
        y = df['Label'].values
        
        # Select only the mapped features
        features_to_use = []
        for cicids_col, suricata_col in feature_mapping.items():
            if cicids_col in df.columns:
                features_to_use.append(cicids_col)
        
        X = df[features_to_use].copy()
        
        # Rename columns to match Suricata feature names
        X.columns = [feature_mapping[col] for col in X.columns]
        
        logger.info(f"Features prepared. Shape: {X.shape}")
        
        return X, y
    
    def _train_decision_tree(self, X_train, y_train, X_test, y_test):
        """Train a Decision Tree classifier."""
        logger.info("Training Decision Tree model...")
        
        # Train model
        dt = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
        dt.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = dt.predict(X_test)
        logger.info("Decision Tree evaluation:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        return dt
    
    def _train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train a Random Forest classifier."""
        logger.info("Training Random Forest model...")
        
        # Train model with regularization parameters
        rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = rf.predict(X_test)
        logger.info("Random Forest evaluation:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        return rf
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test, label_encoder):
        """Train an XGBoost classifier."""
        logger.info("Training XGBoost model...")
        
        # Basic XGBoost model first
        xgb = XGBClassifier(n_estimators=50, random_state=42, scale_pos_weight=5)
        xgb.fit(X_train, y_train)
        
        # Evaluate basic model
        y_pred = xgb.predict(X_test)
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        
        logger.info("XGBoost basic model evaluation:")
        logger.info("\n" + classification_report(y_test_decoded, y_pred_decoded))
        
        # Hyperparameter tuning
        logger.info("Performing hyperparameter tuning for XGBoost...")
        
        param_grid = {
            'max_depth': [6, 9, 12],
            'n_estimators': [100, 150],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        
        # Use GridSearchCV with reduced CV to save time
        grid_search = GridSearchCV(
            estimator=XGBClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,  # Reduced CV for faster training
            scoring='f1_macro',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
        
        # Get best model
        xgb_best = grid_search.best_estimator_
        
        # Evaluate best model
        y_pred = xgb_best.predict(X_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        
        logger.info("XGBoost tuned model evaluation:")
        logger.info("\n" + classification_report(y_test_decoded, y_pred_decoded))
        
        return xgb_best
    
    def _create_baseline(self, X_normal):
        """Create statistical baseline from normal traffic."""
        logger.info("Creating statistical baseline...")
        
        baseline = {}
        for feature in X_normal.columns:
            baseline[feature] = {
                'mean': float(X_normal[feature].mean()),
                'std': float(X_normal[feature].std()),
                'min': float(X_normal[feature].min()),
                'max': float(X_normal[feature].max()),
                'q1': float(X_normal[feature].quantile(0.25)),
                'median': float(X_normal[feature].median()),
                'q3': float(X_normal[feature].quantile(0.75)),
                'iqr': float(X_normal[feature].quantile(0.75) - X_normal[feature].quantile(0.25))
            }
        
        return baseline
    
    def _save_models(self, dt_model, rf_model, xgb_model, scaler, label_encoder, baseline):
        """Save trained models and related files."""
        logger.info(f"Saving models to {self.model_dir}...")
        
        # Save models
        with open(os.path.join(self.model_dir, 'dt_model.pkl'), 'wb') as f:
            pickle.dump(dt_model, f)
        
        with open(os.path.join(self.model_dir, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(rf_model, f)
        
        # Save XGBoost model
        with open(os.path.join(self.model_dir, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(xgb_model, f)
        
        # Save label encoder for XGBoost
        with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Save scaler
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature list
        with open(os.path.join(self.model_dir, 'features.json'), 'w') as f:
            json.dump(ALIGNED_FEATURES, f)
        
        # Save baseline
        with open(os.path.join(self.model_dir, 'baseline.json'), 'w') as f:
            json.dump(baseline, f, indent=4)
        
        logger.info("Models and related files saved successfully.")
    
    def process_suricata_event(self, event):
        """
        Process a Suricata event and update session information.
        
        Args:
            event: Parsed Suricata event object
            
        Returns:
            Finalized session if the event caused a session to be finalized, None otherwise
        """
        if not event:
            return None
        # Whitelist check for events
        try:
            # Skip processing for trusted internal devices like pfSense
            if hasattr(event, 'saddr') and event.saddr in self.service_whitelist.pfsense_interfaces:
                return None
                
            # Extract relevant information for whitelist checking
            if hasattr(event, 'daddr') and hasattr(event, 'dport') and hasattr(event, 'proto'):
                try:
                    dport = int(event.dport) if event.dport else 0
                except (ValueError, TypeError):
                    dport = 0
                    
                if dport > 0 and self.service_whitelist.is_whitelisted(event.daddr, dport, event.proto):
                    logger.debug(f"Skipping whitelisted service: {event.daddr}:{dport} ({event.proto})")
                    return None
                
                # Check for SSH events and log them for debugging
                if dport == 22 or (hasattr(event, 'appproto') and event.appproto == 'ssh'):
                    logger.debug(f"Processing SSH event: {event.saddr} -> {event.daddr}:{dport}")
                
        except Exception as e:
            logger.debug(f"Error in whitelist check: {e}")
            
        # Process event through session manager
        finalized_session = self.session_manager.process_event(event)
        
        # If session was finalized, process it
        if finalized_session:
            # Run finalizer on the session
            result = self.flow_finalizer.process_session(finalized_session)
            
            # Process through behavioral analyzer
            behavioral_features = self.behavioral_analyzer.process_session(finalized_session)
            
            # If behavioral analysis detects anomalies, add to result
            if behavioral_features:
                result['behavioral_features'] = behavioral_features
                
                # Increase anomaly score based on behavioral analysis
                if result['combined_score'] < 0.8 and behavioral_features.get('overall_anomaly_score', 0) > 0.7:
                    result['combined_score'] = 0.8
                    result['is_anomalous'] = True
                    
                    # Log the behavioral alert
                    logger.warning(f"Behavioral anomaly detected for IP {finalized_session.saddr}")
                    
                    # Generate additional alert if not already done
                    if not result.get('is_anomalous'):
                        self.handle_alert(result)
            
            return result
        
        return None
        
    def handle_alert(self, alert_data):
        """
        Handle an alert generated by anomaly detection.
        
        Args:
            alert_data: Dictionary with alert information
        """
        try:
            # Format alert message
            alert_message = self.format_alert(alert_data)
            
            # Log to console
            self._log_alert(alert_data)
            
            # Send to output file if specified
            if hasattr(self, 'output_file') and self.output_file:
                try:
                    # Use UTF-8 encoding to handle all Unicode characters
                    with open(self.output_file, 'a', encoding='utf-8') as f:
                        f.write(f"{alert_message}\n\n")
                except UnicodeEncodeError:
                    # Fallback to ASCII-only version if UTF-8 encoding fails
                    ascii_message = alert_message.encode('ascii', 'replace').decode('ascii')
                    with open(self.output_file, 'a') as f:
                        f.write(f"{ascii_message}\n\n")
                    logger.warning("Unicode characters were replaced in the alert message written to file")
            
            # Send Telegram alert if enabled
            if self.telegram_enabled and self.alerter:
                try:
                    # Let the alerter handle the message sending
                    result = self.alerter.send_message(alert_message)
                    if result:
                        logger.info("Telegram alert sent successfully")
                    else:
                        logger.warning("Failed to send Telegram alert")
                except Exception as e:
                    logger.error(f"Error sending Telegram alert: {e}")
        
        except Exception as e:
            logger.error(f"Error handling alert: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        
    def format_alert(self, alert_data):
        """Format an alert message with detailed information."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            session = alert_data.get('session', {})
            
            message = f"⚠️ ANOMALY DETECTED ⚠️\n"
            message += f"Time: {timestamp}\n"
            message += "-" * 40 + "\n"
            
            # Connection details
            message += "CONNECTION DETAILS:\n"
            message += f"Source IP: {alert_data.get('src_ip', 'Unknown')}\n"
            message += f"Source Port: {alert_data.get('src_port', 'Unknown')}\n"
            message += f"Destination IP: {alert_data.get('dst_ip', 'Unknown')}\n"
            message += f"Destination Port: {alert_data.get('dst_port', 'Unknown')}\n"
            message += f"Protocol: {alert_data.get('proto', 'Unknown')}\n"
            
            # Add application protocol if available
            if 'app_proto' in alert_data and alert_data['app_proto']:
                message += f"App Protocol: {alert_data.get('app_proto', 'Unknown')}\n"
            
            # Add SSH-specific details if it's an SSH connection
            is_ssh = (alert_data.get('app_proto') == 'ssh' or alert_data.get('dst_port') == '22')
            if is_ssh:
                message += f"SSH Authentication Failure: {alert_data.get('ssh_auth_failure', False)}\n"
                if 'recent_ssh_failures' in alert_data:
                    message += f"Recent SSH Auth Failures: {alert_data.get('recent_ssh_failures', 0)}\n"
            
            # Add application layer details if available
            if session.get('http_event_count', 0) > 0:
                message += f"HTTP Events: {session.get('http_event_count', 0)}\n"
                if session.get('http_methods', []):
                    message += f"HTTP Methods: {', '.join(session.get('http_methods', []))}\n"
                if session.get('http_status_codes', []):
                    message += f"HTTP Status Codes: {', '.join(session.get('http_status_codes', []))}\n"
            
            if session.get('dns_event_count', 0) > 0:
                message += f"DNS Events: {session.get('dns_event_count', 0)}\n"
                if session.get('dns_queries', []):
                    queries = session.get('dns_queries', [])[:3]  # Show only first 3
                    message += f"DNS Queries: {', '.join(queries)}\n"
            
            if session.get('tls_event_count', 0) > 0:
                message += f"TLS Events: {session.get('tls_event_count', 0)}\n"
                if session.get('tls_sni', []):
                    message += f"TLS SNI: {', '.join(session.get('tls_sni', []))}\n"
            
            # Add flow timing information
            if 'starttime' in session:
                message += f"Flow Start: {session.get('starttime', 'Unknown')}\n"
            if 'endtime' in session:
                message += f"Flow End: {session.get('endtime', 'Unknown')}\n"
            if 'duration' in alert_data:
                message += f"Duration: {float(alert_data.get('duration', 0)):.3f} seconds\n"
            
            # Traffic volume stats
            message += "-" * 40 + "\n"
            message += "TRAFFIC STATISTICS:\n"
            
            try:
                total_bytes = alert_data.get('total_bytes', 0)
                total_packets = alert_data.get('total_packets', 0)
                fwd_bytes = session.get('total_fwd_bytes', 0)
                bwd_bytes = session.get('total_bwd_bytes', 0)
                fwd_packets = session.get('total_fwd_packets', 0)
                bwd_packets = session.get('total_bwd_packets', 0)
                
                message += f"Total Bytes: {total_bytes:,}\n"
                message += f"  → Source→Dest: {fwd_bytes:,} bytes\n"
                message += f"  → Dest→Source: {bwd_bytes:,} bytes\n"
                message += f"Total Packets: {total_packets:,}\n"
                message += f"  → Source→Dest: {fwd_packets:,} packets\n"
                message += f"  → Dest→Source: {bwd_packets:,} packets\n"
            except Exception as e:
                message += f"Error processing traffic statistics: {str(e)}\n"
            
            # Connection state
            if 'state' in session:
                message += f"Connection State: {session.get('state', '')}\n"
            
            # Anomaly detection results
            message += "-" * 40 + "\n"
            message += "ANOMALY DETECTION RESULTS:\n"
            message += f"Overall Anomaly Score: {alert_data.get('combined_score', 0):.4f}\n"
            
            # Add ML detection details
            ml_result = alert_data.get('ml_result', {})
            stat_result = alert_data.get('stat_result', {})
            
            message += "\nMachine Learning Detection:\n"
            
            # Decision Tree
            if 'dt_confidence' in ml_result:
                message += f"  → Decision Tree: {'**Anomalous**' if ml_result.get('dt_prediction') == 1 else 'Benign'} (confidence: {ml_result.get('dt_confidence', 0):.2f})\n"
            else:
                message += f"  → Decision Tree: {'**Anomalous**' if ml_result.get('dt_prediction') == 1 else 'Benign'}\n"
                
            # Random Forest
            if 'rf_confidence' in ml_result:
                message += f"  → Random Forest: {'**Anomalous**' if ml_result.get('rf_prediction') == 1 else 'Benign'} (confidence: {ml_result.get('rf_confidence', 0):.2f})\n"
            else:
                message += f"  → Random Forest: {'**Anomalous**' if ml_result.get('rf_prediction') == 1 else 'Benign'}\n"
                
            # XGBoost (if available)
            if 'xgb_prediction' in ml_result:
                if 'xgb_confidence' in ml_result:
                    message += f"  → XGBoost: {'**Anomalous**' if ml_result.get('xgb_prediction') == 1 else 'Benign'} (confidence: {ml_result.get('xgb_confidence', 0):.2f})\n"
                else:
                    message += f"  → XGBoost: {'**Anomalous**' if ml_result.get('xgb_prediction') == 1 else 'Benign'}\n"
            
            # Add statistical anomaly details
            if stat_result.get('details') and len(stat_result.get('details', [])) > 0:
                message += "\nStatistical Anomalies:\n"
                for detail in stat_result.get('details', [])[:5]:  # Show top 5 anomalous features
                    try:
                        feature = detail.get('feature', 'Unknown')
                        value = detail.get('value', 0)
                        z_score = detail.get('z_score', 0)
                        is_outlier = detail.get('is_outlier', False)
                        
                        if feature == 'app_layer':
                            message += f"  → Application layer anomalies detected (score: {value:.2f})\n"
                            continue
                            
                        mean = detail.get('baseline_mean', 0)
                        std = detail.get('baseline_std', 1)
                        
                        # Format the anomaly description
                        if is_outlier:
                            outlier_type = "high" if value > mean else "low"
                            message += f"  → {feature}: {value:.2f} is an {outlier_type} outlier (z-score: {z_score:.2f})\n"
                            message += f"     Normal range: {mean-2*std:.2f} to {mean+2*std:.2f}, mean: {mean:.2f}\n"
                        else:
                            message += f"  → {feature}: {value:.2f} is anomalous (z-score: {z_score:.2f})\n"
                            message += f"     Normal value: {mean:.2f} ± {std:.2f}\n"
                    except Exception as e:
                        message += f"  → Error formatting anomaly detail: {str(e)}\n"
            
            # Add behavioral anomaly details if available
            behavioral_features = alert_data.get('behavioral_features', {})
            if behavioral_features:
                message += "\nBehavioral Analysis:\n"
                
                # Add scan detection
                if behavioral_features.get('port_scan_score', 0) > 0.5:
                    message += f"  → Port scan activity detected (score: {behavioral_features.get('port_scan_score', 0):.2f})\n"
                    message += f"     Scanned {behavioral_features.get('unique_dst_ports', 0)} ports\n"
                
                if behavioral_features.get('host_scan_score', 0) > 0.5:
                    message += f"  → Host scan activity detected (score: {behavioral_features.get('host_scan_score', 0):.2f})\n"
                    message += f"     Scanned {behavioral_features.get('unique_dst_ips', 0)} hosts\n"
                
                # Add brute force detection
                if behavioral_features.get('brute_force_score', 0) > 0.5:
                    message += f"  → Brute force activity detected (score: {behavioral_features.get('brute_force_score', 0):.2f})\n"
                    message += f"     Auth failures: {behavioral_features.get('auth_failures_per_second', 0):.2f} per second\n"
                    
                # Add SSH-specific metrics
                if is_ssh and 'ssh_auth_failures' in behavioral_features:
                    message += f"  → SSH Auth Failures: {behavioral_features.get('ssh_auth_failures', 0)}\n"
                    message += f"     Rate: {behavioral_features.get('ssh_auth_failures_per_second', 0):.2f} per second\n"
                
                # Add volume anomalies
                if behavioral_features.get('volume_anomaly_score', 0) > 0.5:
                    message += f"  → Traffic volume anomaly (score: {behavioral_features.get('volume_anomaly_score', 0):.2f})\n"
                    message += f"     Bytes per second: {behavioral_features.get('bytes_sent_per_second', 0):.2f}\n"
                    message += f"     Packets per second: {behavioral_features.get('packets_sent_per_second', 0):.2f}\n"
            
            message += "-" * 40 + "\n"
            
            # Add possible threat implications
            message += "POSSIBLE THREAT IMPLICATIONS:\n"
            
            # Some basic threat heuristics
            try:
                port = int(alert_data.get('dst_port', 0) or 0)
                
                if port == 22:
                    message += "  • SSH brute force attempt or unauthorized access\n"
                    if 'recent_ssh_failures' in alert_data and alert_data['recent_ssh_failures'] > 2:
                        message += f"  • Multiple SSH auth failures ({alert_data['recent_ssh_failures']}) detected - strong indication of brute force\n"
                elif port == 23:
                    message += "  • Telnet activity - insecure protocol potentially indicating compromise\n"
                elif port == 3389:
                    message += "  • RDP connection - potential remote access activity\n"
                elif port in [80, 443]:
                    message += "  • HTTP/HTTPS traffic with unusual patterns - potential web attack or data exfiltration\n"
                elif port == 53:
                    message += "  • Unusual DNS traffic - potential DNS tunneling or C2 communication\n"
                elif port == 445 or port == 139:
                    message += "  • SMB/NetBIOS traffic - potential lateral movement or file access\n"
                elif port < 1024:
                    message += "  • Well-known service port with anomalous behavior\n"
                elif port > 49000:
                    message += "  • High port communication - potential backdoor or non-standard service\n"
                else:
                    message += "  • Unusual network traffic patterns detected\n"
            except (ValueError, TypeError):
                message += "  • Unusual network traffic patterns detected\n"
            
            # Add behavioral implications
            if behavioral_features:
                if behavioral_features.get('port_scan_score', 0) > 0.5:
                    message += "  • Port scanning may indicate reconnaissance activity\n"
                
                if behavioral_features.get('host_scan_score', 0) > 0.5:
                    message += "  • Host scanning may indicate lateral movement attempts\n"
                
                if behavioral_features.get('brute_force_score', 0) > 0.5:
                    message += "  • Authentication failures may indicate brute force attacks\n"
                
                if behavioral_features.get('volume_anomaly_score', 0) > 0.5:
                    message += "  • High volume traffic may indicate DoS, data exfiltration, or malware activity\n"
            
            # Zero-byte flow pattern
            if alert_data.get('zero_byte_pattern', False):
                message += "  • Repeated zero-byte flows may indicate scanning, brute force, or C2 beaconing\n"
                
            # Add timestamp
            message += f"\nAlert generated: {timestamp}\n"
            
            return message
        except Exception as e:
            logger.error(f"Error formatting alert: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a basic alert if we encounter an error
            return f"ANOMALY DETECTED\nScore: {alert_data.get('combined_score', 0):.4f}\nSource: {alert_data.get('src_ip', 'Unknown')}:{alert_data.get('src_port', 'Unknown')}\nDestination: {alert_data.get('dst_ip', 'Unknown')}:{alert_data.get('dst_port', 'Unknown')}"
    
    def _log_alert(self, alert_data):
        """Log an alert to the console in a readable format."""
        print("\n" + "!" * 80)
        logger.info("ALERT DETECTED")
        
        # Basic connection info
        logger.info(f"Source: {alert_data.get('src_ip', 'Unknown')}:{alert_data.get('src_port', 'Unknown')}")
        logger.info(f"Destination: {alert_data.get('dst_ip', 'Unknown')}:{alert_data.get('dst_port', 'Unknown')}")
        logger.info(f"Protocol: {alert_data.get('proto', 'Unknown')}")
        logger.info(f"App Protocol: {alert_data.get('app_proto', 'Unknown')}")
        
        # Session information
        session = alert_data.get('session', {})
        logger.info(f"Flow ID: {session.get('flow_id', 'Unknown')}")
        logger.info(f"Duration: {alert_data.get('duration', 0):.3f} seconds")
        
        # App layer info
        if session.get('http_event_count', 0) > 0:
            logger.info(f"HTTP: {session.get('http_event_count', 0)} events")
        if session.get('dns_event_count', 0) > 0:
            logger.info(f"DNS: {session.get('dns_event_count', 0)} events")
        if session.get('tls_event_count', 0) > 0:
            logger.info(f"TLS: {session.get('tls_event_count', 0)} events")
        if session.get('ssh_event_count', 0) > 0:
            logger.info(f"SSH: {session.get('ssh_event_count', 0)} events")
            # Check for SSH auth failures
            if session.get('ssh_auth_failure', False):
                logger.info("SSH AUTH FAILURE DETECTED")
        
        # Traffic volume
        logger.info(f"Bytes: {alert_data.get('total_bytes', 0):,}, Packets: {alert_data.get('total_packets', 0):,}")
        
        # Alert scores
        logger.info(f"Anomaly Score: {alert_data.get('combined_score', 0):.4f}")
        
        # Special cases
        if alert_data.get('zero_byte_pattern', False):
            logger.info("ZERO-BYTE PATTERN DETECTED - Possible scanning or brute force")
        
        # SSH-specific info
        if 'is_ssh' in alert_data and alert_data['is_ssh']:
            logger.info("SSH CONNECTION DETECTED")
            if 'ssh_auth_failure' in alert_data and alert_data['ssh_auth_failure']:
                logger.info("SSH AUTHENTICATION FAILURE")
            if 'recent_ssh_failures' in alert_data:
                logger.info(f"Recent SSH Auth Failures: {alert_data['recent_ssh_failures']}")
        
        # Behavioral information
        behavioral_features = alert_data.get('behavioral_features', {})
        if behavioral_features:
            logger.info("Behavioral Analysis:")
            
            if behavioral_features.get('port_scan_score', 0) > 0.5:
                logger.info(f"  - Port scan score: {behavioral_features.get('port_scan_score', 0):.2f}")
            
            if behavioral_features.get('host_scan_score', 0) > 0.5:
                logger.info(f"  - Host scan score: {behavioral_features.get('host_scan_score', 0):.2f}")
            
            if behavioral_features.get('brute_force_score', 0) > 0.5:
                logger.info(f"  - Brute force score: {behavioral_features.get('brute_force_score', 0):.2f}")
            
            if behavioral_features.get('volume_anomaly_score', 0) > 0.5:
                logger.info(f"  - Volume anomaly score: {behavioral_features.get('volume_anomaly_score', 0):.2f}")
            
            logger.info(f"  - Overall behavioral score: {behavioral_features.get('overall_anomaly_score', 0):.2f}")
        
        print("!" * 80)
    
    def analyze_suricata_file(self, file_path, output_file=None):
        """Analyze a Suricata JSON log file with session and behavior awareness."""
        logger.info(f"Analyzing Suricata JSON log file: {file_path}")
        self.output_file = output_file
        
        # Initialize counters
        total_entries = 0
        flagged_by_suricata = 0
        processed_events = 0
        finalized_sessions = 0
        anomalies_detected = 0
        ssh_sessions = 0
        
        try:
            # Process the file line by line
            with open(file_path, 'r') as f:
                for line in f:
                    total_entries += 1
                    
                    try:
                        # Parse JSON
                        entry = json.loads(line)
                        
                        # Check if already flagged by Suricata
                        if 'alert' in entry:
                            flagged_by_suricata += 1
                            continue
                        
                        # Check if this is an SSH event (for tracking)
                        is_ssh = False
                        if 'dest_port' in entry and entry['dest_port'] == 22:
                            is_ssh = True
                            ssh_sessions += 1
                            
                        # Process entry using parser
                        event = self.parser.process_line(entry)
                        
                        if not event:
                            continue
                        
                        processed_events += 1
                        
                        # Process through session manager
                        result = self.process_suricata_event(event)
                        
                        if result:
                            finalized_sessions += 1
                            if result.get('is_anomalous', False):
                                anomalies_detected += 1
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line: {line[:50]}...")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing entry: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue
                    
                    # Periodically clean up session and behavioral data (more frequently)
                    if total_entries % 500 == 0:  # Check more often (every 500 entries)
                        # Clean up expired sessions
                        expired_sessions = self.session_manager.cleanup_expired_sessions()
                        
                        # Process any expired sessions
                        for session in expired_sessions:
                            result = self.flow_finalizer.process_session(session)
                            finalized_sessions += 1
                            if result.get('is_anomalous', False):
                                anomalies_detected += 1
                        
                        # Clean up behavioral analyzer
                        self.behavioral_analyzer.cleanup()
                        
                        # Print progress update
                        if total_entries % 5000 == 0:  # Less frequent updates to console
                            logger.info(f"Processed {total_entries} entries, {finalized_sessions} finalized sessions, {anomalies_detected} anomalies, {ssh_sessions} SSH sessions")
            
            # Final cleanup of any remaining sessions
            expired_sessions = self.session_manager.cleanup_expired_sessions()
            for session in expired_sessions:
                result = self.flow_finalizer.process_session(session)
                finalized_sessions += 1
                if result.get('is_anomalous', False):
                    anomalies_detected += 1
            
            # Print summary
            logger.info(f"\nAnalysis Summary:")
            logger.info(f"Total log entries: {total_entries}")
            logger.info(f"Entries flagged by Suricata: {flagged_by_suricata}")
            logger.info(f"Processed events: {processed_events}")
            logger.info(f"Finalized sessions: {finalized_sessions}")
            logger.info(f"SSH sessions: {ssh_sessions}")
            logger.info(f"Anomalies detected: {anomalies_detected}")
            logger.info(f"Session Manager stats: {self.session_manager.get_stats()}")
            logger.info(f"Behavioral Analyzer stats: {self.behavioral_analyzer.get_stats()}")
            logger.info(f"Flow Finalizer stats: {self.flow_finalizer.get_stats()}")
        except KeyboardInterrupt:
                # Print summary when stopped
                logger.info("\n\nAnalysis stopped.")
                logger.info(f"Total entries: {total_entries}")
                logger.info(f"Entries flagged by Suricata: {flagged_by_suricata}")
                logger.info(f"Processed events: {processed_events}")
                logger.info(f"Finalized sessions: {finalized_sessions}")
                logger.info(f"SSH sessions: {ssh_sessions}")
                logger.info(f"Anomalies detected: {anomalies_detected}")
                logger.info(f"Session Manager stats: {self.session_manager.get_stats()}")
                logger.info(f"Behavioral Analyzer stats: {self.behavioral_analyzer.get_stats()}")
                logger.info(f"Flow Finalizer stats: {self.flow_finalizer.get_stats()}")
            
    def monitor_suricata_file(self, file_path, output_file=None):
        """Monitor a Suricata JSON log file in real-time with session and behavior awareness."""
        logger.info(f"Monitoring Suricata JSON log file: {file_path}")
        self.output_file = output_file
        
        # Get initial file position
        with open(file_path, 'r') as f:
            f.seek(0, 2)  # Move to end of file
            position = f.tell()
        
        # Initialize counters
        total_entries = 0
        flagged_by_suricata = 0
        processed_events = 0
        finalized_sessions = 0
        anomalies_detected = 0
        ssh_sessions = 0
        last_cleanup = time.time()
        last_stats_update = time.time()
        
        # More frequent cleanups for SSH sessions
        ssh_cleanup_interval = 5  # Check SSH sessions every 5 seconds
        last_ssh_cleanup = time.time()
        
        try:
            while True:
                # Check if file has grown
                with open(file_path, 'r') as f:
                    f.seek(0, 2)
                    end_position = f.tell()
                    
                    if end_position > position:
                        # Process new data
                        f.seek(position)
                        for line in f:
                            total_entries += 1
                            
                            try:
                                # Parse JSON
                                entry = json.loads(line)
                                
                                # Check if already flagged by Suricata
                                if 'alert' in entry:
                                    flagged_by_suricata += 1
                                    continue
                                
                                # Check if this is an SSH event
                                is_ssh = False
                                if 'dest_port' in entry and str(entry['dest_port']) == '22':
                                    is_ssh = True
                                    ssh_sessions += 1
                                    
                                # Process entry using parser
                                event = self.parser.process_line(entry)
                                
                                if not event:
                                    continue
                                
                                processed_events += 1
                                
                                # Process through session manager
                                result = self.process_suricata_event(event)
                                
                                if result:
                                    finalized_sessions += 1
                                    if result.get('is_anomalous', False):
                                        anomalies_detected += 1
                            
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                logger.error(f"Error processing entry: {e}")
                                continue
                        
                        # Update position
                        position = end_position
                
                # More frequent checks for SSH sessions
                current_time = time.time()
                if current_time - last_ssh_cleanup > ssh_cleanup_interval:
                    # Check for SSH sessions that need to be finalized
                    ssh_sessions = self.session_manager.get_sessions_by_dest_port(22)
                    if ssh_sessions:
                        for ssh_session in ssh_sessions:
                            if ssh_session.needs_immediate_processing:
                                logger.debug(f"Processing SSH session that needs immediate analysis")
                                ssh_session.finalize()
                                result = self.flow_finalizer.process_session(ssh_session)
                                finalized_sessions += 1
                                if result.get('is_anomalous', False):
                                    anomalies_detected += 1
                    
                    last_ssh_cleanup = current_time
                
                # Check if regular cleanup is needed
                if current_time - last_cleanup > 15:  # Cleanup every 15 seconds (more frequent)
                    # Clean up expired sessions
                    expired_sessions = self.session_manager.cleanup_expired_sessions()
                    
                    # Process any expired sessions
                    for session in expired_sessions:
                        result = self.flow_finalizer.process_session(session)
                        finalized_sessions += 1
                        if result.get('is_anomalous', False):
                            anomalies_detected += 1
                    
                    # Clean up behavioral analyzer
                    self.behavioral_analyzer.cleanup()
                    
                    last_cleanup = current_time
                
                # Print status update every minute
                if current_time - last_stats_update > 60:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(f"[{now}] Processed: {processed_events} | Sessions: {finalized_sessions} | Alerts: {anomalies_detected} | SSH: {ssh_sessions}")
                    
                    # Check for suspicious IPs based on behavioral analysis
                    top_anomalous = self.behavioral_analyzer.get_top_anomalous_ips(5)
                    if top_anomalous:
                        logger.info("Top suspicious IPs from behavioral analysis:")
                        for ip, score in top_anomalous:
                            if score > 0.5:  # Lower threshold to highlight more suspicious IPs
                                logger.info(f"  - {ip}: Score {score:.2f}")
                    
                    # Check for brute force activity
                    brute_force_ips = self.behavioral_analyzer.get_brute_force_activity(0.3)  # Lower threshold
                    if brute_force_ips:
                        logger.info("Potential brute force activity:")
                        for ip, score in brute_force_ips:
                            logger.info(f"  - {ip}: Score {score:.2f}")
                    
                    last_stats_update = current_time
                
                # Sleep for a short time
                time.sleep(0.5)  # Faster polling (0.5s instead of 1s)
        
        except KeyboardInterrupt:
            # Print summary when stopped
            logger.info("\n\nMonitoring stopped.")
            logger.info(f"Total entries: {total_entries}")
            logger.info(f"Entries flagged by Suricata: {flagged_by_suricata}")
            logger.info(f"Processed events: {processed_events}")
            logger.info(f"Finalized sessions: {finalized_sessions}")
            logger.info(f"SSH sessions: {ssh_sessions}")
            logger.info(f"Anomalies detected: {anomalies_detected}")
            logger.info(f"Session Manager stats: {self.session_manager.get_stats()}")
            logger.info(f"Behavioral Analyzer stats: {self.behavioral_analyzer.get_stats()}")
            logger.info(f"Flow Finalizer stats: {self.flow_finalizer.get_stats()}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Hybrid NIDS with Session and Behavioral Awareness')
    
    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--train', 
                             help='Train models using CICIDS2017 dataset')
    action_group.add_argument('--analyze', 
                             help='Analyze a Suricata JSON log file')
    action_group.add_argument('--realtime', 
                             help='Monitor a Suricata JSON log file in real-time')
    
    # Other arguments
    parser.add_argument('--model_dir', 
                       default=os.getenv('MODEL_DIR', './model'),
                       help='Directory containing trained models')
    parser.add_argument('--output', 
                       help='Output file for alerts')
    parser.add_argument('--telegram', 
                       action='store_true',
                       help='Enable Telegram alerts')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Initialize Hybrid NIDS
    nids = HybridNIDS(model_dir=args.model_dir, telegram_enabled=args.telegram)
    
    # Execute the selected action
    if args.train:
        nids.train(args.train)
    elif args.analyze:
        nids.analyze_suricata_file(args.analyze, args.output)
    elif args.realtime:
        nids.monitor_suricata_file(args.realtime, args.output)
    
    # No need to explicitly disconnect the Telegram client
    # The daemon thread will be terminated when the main program exits
    logger.info("NIDS execution completed")


if __name__ == "__main__":
    main()