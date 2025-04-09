#!/usr/bin/env python3
"""
Hybrid NIDPS - Detection Module (Revised Version)

This script processes Suricata JSON logs, detects anomalies, and alerts administrators.
Uses SLIPS flow analysis logic for proper flow parsing.

Usage:
    python detect.py --suricata <path_to_suricata_json> --model_dir <model_directory> 
                    --telegram_token <token> --telegram_chat_id <chat_id>
                    [--mode analyze|monitor]
"""

import argparse
import os
import sys
import json
import time
import datetime
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from dataclasses import asdict
from telegram_alert import TelegramAlerter
from suricata_parser import SuricataParser
from flow_features import FlowFeatureExtractor

# Load environment variables from .env file
load_dotenv()

def parse_args():
    """Parse command-line arguments with environment variable fallbacks."""
    parser = argparse.ArgumentParser(description='Hybrid NIDPS - Detection Module')
    parser.add_argument('--suricata', 
                        default=os.getenv('DEFAULT_SURICATA_FILE'),
                        help='Path to Suricata JSON log file')
    parser.add_argument('--model_dir', 
                        default=os.getenv('MODEL_DIR', './model'),
                        help='Directory containing trained models')
    
    # Telegram arguments
    telegram_group = parser.add_argument_group('Telegram Options')
    telegram_group.add_argument('--telegram_token', 
                               default=os.getenv('TELEGRAM_TOKEN'),
                               help='Telegram bot token')
    telegram_group.add_argument('--telegram_chat_id', 
                               default=os.getenv('TELEGRAM_CHAT_ID'),
                               help='Telegram chat ID')
    telegram_group.add_argument('--telegram_api_id', 
                               default=os.getenv('TELEGRAM_API_ID'),
                               help='Telegram API ID (for Telethon)')
    telegram_group.add_argument('--telegram_api_hash', 
                               default=os.getenv('TELEGRAM_API_HASH'),
                               help='Telegram API hash (for Telethon)')
    
    parser.add_argument('--mode', choices=['analyze', 'monitor'], 
                        default=os.getenv('DEFAULT_MODE', 'analyze'),
                        help='Operating mode: analyze (static file) or monitor (real-time)')
    
    args = parser.parse_args()
    
    # Validate required parameters
    if not args.suricata:
        parser.error("Suricata JSON file path is required. Provide it via --suricata or set DEFAULT_SURICATA_FILE in .env")
    
    if not args.telegram_token:
        parser.error("Telegram token is required. Provide it via --telegram_token or set TELEGRAM_TOKEN in .env")
    
    if not args.telegram_chat_id:
        parser.error("Telegram chat ID is required. Provide it via --telegram_chat_id or set TELEGRAM_CHAT_ID in .env")
    
    return args

def load_models(model_dir):
    """Load trained models and related files."""
    print("Loading models and configuration...")
    
    result = {}
    
    # Load Decision Tree model
    with open(os.path.join(model_dir, 'dt_model.pkl'), 'rb') as f:
        result['dt_model'] = pickle.load(f)
    
    # Load Random Forest model
    with open(os.path.join(model_dir, 'rf_model.pkl'), 'rb') as f:
        result['rf_model'] = pickle.load(f)
    
    # Load selected features
    with open(os.path.join(model_dir, 'selected_features.json'), 'r') as f:
        result['selected_features'] = json.load(f)
    
    # Load baseline
    with open(os.path.join(model_dir, 'baseline.json'), 'r') as f:
        result['baseline'] = json.load(f)
    
    print(f"Loaded models with {len(result['selected_features'])} selected features.")
    
    return result

def detect_ml_anomaly(features, models):
    """Detect anomalies using machine learning models."""
    # Get predictions from both models
    dt_pred = models['dt_model'].predict(features)[0]
    rf_pred = models['rf_model'].predict(features)[0]
    
    # Get prediction probabilities
    dt_proba = models['dt_model'].predict_proba(features)[0]
    rf_proba = models['rf_model'].predict_proba(features)[0]
    
    # Calculate anomaly score
    dt_score = dt_proba[1] if len(dt_proba) > 1 else 0
    rf_score = rf_proba[1] if len(rf_proba) > 1 else 0
    
    # Combine scores
    combined_score = (dt_score + rf_score) / 2
    
    # Determine if anomalous (if either model predicts anomaly)
    is_anomalous = dt_pred == 1 or rf_pred == 1
    
    return {
        'score': combined_score,
        'is_anomalous': is_anomalous,
        'dt_prediction': dt_pred,
        'rf_prediction': rf_pred
    }

def detect_statistical_anomaly(features, baseline):
    """Detect anomalies using statistical methods."""
    # Initialize anomaly score and details
    anomaly_score = 0
    anomaly_details = []
    
    # Check each feature against the baseline
    for feature in features.columns:
        if feature not in baseline:
            continue
        
        # Get feature value
        value = features[feature].values[0]
        
        # Get baseline stats
        stats = baseline[feature]
        
        # Calculate Z-score if std > 0
        if stats['std'] > 0:
            z_score = (value - stats['mean']) / stats['std']
        else:
            z_score = 0
        
        # Check for outlier based on IQR
        iqr = stats['iqr']
        if iqr > 0:
            lower_bound = stats['q1'] - 1.5 * iqr
            upper_bound = stats['q3'] + 1.5 * iqr
            is_outlier = value < lower_bound or value > upper_bound
        else:
            is_outlier = False
        
        # Add to anomaly score
        anomaly_score += abs(z_score) if abs(z_score) > 3 else 0
        
        # Add to details if anomalous
        if abs(z_score) > 3 or is_outlier:
            anomaly_details.append({
                'feature': feature,
                'value': value,
                'z_score': z_score,
                'is_outlier': is_outlier,
                'baseline_mean': stats['mean'],
                'baseline_std': stats['std']
            })
    
    # Normalize score
    if len(features.columns) > 0:
        anomaly_score /= len(features.columns)
    
    # Determine if anomalous
    is_anomalous = anomaly_score > 0.5 or len(anomaly_details) >= 3
    
    return {
        'score': anomaly_score,
        'is_anomalous': is_anomalous,
        'details': anomaly_details
    }

def analyze_suricata_file(file_path, models, alerter):
    """Analyze a Suricata JSON log file for anomalies."""
    print(f"Analyzing Suricata JSON log file: {file_path}")
    
    # Initialize counters
    total_entries = 0
    flagged_by_suricata = 0
    processed_entries = 0
    anomalies_detected = 0
    
    # Initialize parser and feature extractor
    parser = SuricataParser()
    feature_extractor = FlowFeatureExtractor(models['selected_features'])
    
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
                
                # Process entry using SLIPS parser
                flow = parser.process_line(entry)
                
                if not flow:
                    continue
                
                processed_entries += 1
                
                # Extract features
                features = feature_extractor.extract_from_flow(flow)
                
                # Skip if we couldn't extract enough features
                if features.empty or features.shape[1] < 5:
                    continue
                    
                # Detect anomalies
                ml_result = detect_ml_anomaly(features, models)
                stat_result = detect_statistical_anomaly(features, models['baseline'])
                
                # Combine results
                is_anomalous = ml_result['is_anomalous'] or stat_result['is_anomalous']
                
                if is_anomalous:
                    anomalies_detected += 1
                    
                    # Convert flow to dict for alerting
                    flow_dict = asdict(flow)
                    
                    # Calculate combined anomaly score
                    anomaly_score = max(ml_result['score'], stat_result['score'])
                    
                    # Generate and send alert
                    alert_message = alerter.format_anomaly_alert(
                        flow_dict, anomaly_score, stat_result['details']
                    )
                    # alerter.send_alert(alert_message)
                    print(alert_message)
                    # Print to console
                    print(f"\nALERT {anomalies_detected}: {flow_dict.get('saddr')}:{flow_dict.get('sport')} -> "
                          f"{flow_dict.get('daddr')}:{flow_dict.get('dport')} ({flow_dict.get('proto')})")
                    print(f"Anomaly Score: {anomaly_score:.4f}")
            
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line[:50]}...")
                continue
            except Exception as e:
                print(f"Error processing entry: {e}")
                continue
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total entries: {total_entries}")
    print(f"Entries flagged by Suricata: {flagged_by_suricata}")
    print(f"Entries processed for anomaly detection: {processed_entries}")
    print(f"Anomalies detected: {anomalies_detected}")

def monitor_suricata_file(file_path, models, alerter):
    """Monitor a Suricata JSON log file in real-time for anomalies."""
    print(f"Monitoring Suricata JSON log file: {file_path}")
    
    # Get initial file position
    with open(file_path, 'r') as f:
        f.seek(0, 2)  # Move to end of file
        position = f.tell()
    
    # Initialize counters
    total_entries = 0
    flagged_by_suricata = 0
    processed_entries = 0
    anomalies_detected = 0
    
    # Initialize parser and feature extractor
    parser = SuricataParser()
    feature_extractor = FlowFeatureExtractor(models['selected_features'])
    
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
                            
                            # Process entry using SLIPS parser
                            flow = parser.process_line(entry)
                            
                            if not flow:
                                continue
                            
                            processed_entries += 1
                            
                            # Extract features
                            features = feature_extractor.extract_from_flow(flow)
                            
                            # Skip if we couldn't extract enough features
                            if features.empty or features.shape[1] < 5:
                                continue
                                
                            # Detect anomalies
                            ml_result = detect_ml_anomaly(features, models)
                            stat_result = detect_statistical_anomaly(features, models['baseline'])
                            
                            # Combine results
                            is_anomalous = ml_result['is_anomalous'] or stat_result['is_anomalous']
                            
                            if is_anomalous:
                                anomalies_detected += 1
                                
                                # Convert flow to dict for alerting
                                flow_dict = asdict(flow)
                                
                                # Calculate combined anomaly score
                                anomaly_score = max(ml_result['score'], stat_result['score'])
                                
                                # Generate and send alert
                                alert_message = alerter.format_anomaly_alert(
                                    flow_dict, anomaly_score, stat_result['details']
                                )
                                # alerter.send_alert(alert_message)
                                print(alert_message)
                                # Print to console
                                print(f"\nALERT {anomalies_detected}: {flow_dict.get('saddr')}:{flow_dict.get('sport')} -> "
                                      f"{flow_dict.get('daddr')}:{flow_dict.get('dport')} ({flow_dict.get('proto')})")
                                print(f"Anomaly Score: {anomaly_score:.4f}")
                        
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"Error processing entry: {e}")
                            continue
                    
                    # Update position
                    position = end_position
            
            # Print status update every minute
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Processed: {processed_entries} | Alerts: {anomalies_detected}", end="\r")
            
            # Sleep for a second
            time.sleep(1)
    
    except KeyboardInterrupt:
        # Print summary when stopped
        print("\n\nMonitoring stopped.")
        print(f"Total entries: {total_entries}")
        print(f"Entries flagged by Suricata: {flagged_by_suricata}")
        print(f"Entries processed for anomaly detection: {processed_entries}")
        print(f"Anomalies detected: {anomalies_detected}")

def main():
    """Main function."""
    args = parse_args()
    
    # Load models
    models = load_models(args.model_dir)
    
    # Initialize Telegram alerter
    alerter = TelegramAlerter()
    print(alerter)
    # Execute the selected mode
    if args.mode == 'analyze':
        analyze_suricata_file(args.suricata, models, alerter)
    elif args.mode == 'monitor':
        monitor_suricata_file(args.suricata, models, alerter)
    else:
        print(f"Error: Unknown mode: {args.mode}")
        sys.exit(1)
        
    # Keep the alerter's client running until program completion
    # This ensures messages have time to be sent
    if hasattr(alerter, 'loop') and alerter.loop and alerter.loop.is_running():
        print("Waiting for pending Telegram operations to complete...")
        import time
        time.sleep(3)  # Give pending operations time to complete

if __name__ == "__main__":
    main()