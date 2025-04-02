#!/usr/bin/env python3
"""
Network Intrusion Detection System - Training Module

This script implements a hybrid Network Intrusion Detection System (NIDS) based on 
machine learning and anomaly detection. It has two modes:
1. train: Trains classification models on CICIDS2017 dataset
2. baseline: Creates a baseline of normal network behavior from eve.json files

Based on the paper "A Suricata and Machine Learning Based Hybrid Network Intrusion Detection System"
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
import time
import pickle
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Set plotting style
sns.set(style='darkgrid')

def load_cicids2017(dataset_path):
    """
    Load and preprocess the CICIDS2017 dataset.
    Based on methodology from nids-ml-hwr.py
    
    Args:
        dataset_path: Path to the CICIDS2017 dataset folder or files
        
    Returns:
        Preprocessed DataFrame
    """
    print("Loading CICIDS2017 dataset...")
    
    # If path is a directory, read all CSV files
    if os.path.isdir(dataset_path):
        df_full = pd.DataFrame()
        for dirname, _, filenames in os.walk(dataset_path):
            for filename in filenames:
                if filename.endswith('.csv'):
                    file_path = os.path.join(dirname, filename)
                    print(f"Reading file: {file_path}")
                    df_full = pd.concat([df_full, pd.read_csv(file_path)], ignore_index=True)
    else:
        # If path is a file, read the single file
        df_full = pd.read_csv(dataset_path)
    
    # Clean column names (remove leading/trailing spaces)
    df_full.columns = df_full.columns.str.strip()
    
    return df_full

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, duplicates, and infinite values.
    Based on methodology from nids-ml-hwr.py
    
    Args:
        df: Input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    print("Preprocessing data...")
    original_shape = df.shape
    
    # Check for infinite values
    is_inf = df.isin([np.inf, -np.inf])
    inf_counts = is_inf.sum()
    columns_with_inf = inf_counts[inf_counts > 0]
    print(f"Columns with infinite values: {len(columns_with_inf)}")
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check for NaN values
    nan_counts = df.isnull().sum()
    columns_with_nan = nan_counts[nan_counts > 0]
    print(f"Columns with NaN values: {len(columns_with_nan)}")
    
    # Remove rows with NaN values
    df.dropna(inplace=True)
    
    # Check and remove duplicate rows
    duplicate_rows_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_rows_count}")
    df.drop_duplicates(inplace=True)
    
    # Report data reduction
    print(f"Original shape: {original_shape}")
    print(f"After preprocessing: {df.shape}")
    print(f"Reduction: {original_shape[0] - df.shape[0]} rows ({(original_shape[0] - df.shape[0])/original_shape[0]*100:.2f}%)")
    
    return df

def map_attack_categories(df, column='Label'):
    """
    Map attack types into broader categories as specified in nids-ml-hwr.py
    
    Args:
        df: DataFrame containing the 'Label' or specified column
        column: Name of the column containing attack labels
        
    Returns:
        DataFrame with an additional 'Attack Type' column
    """
    print("Mapping attack categories...")
    
    # Define the attack mapping as per nids-ml-hwr.py
    attack_map = {
    'BENIGN': 'BENIGN',
    'Bot': 'BOTNET',
    'DDoS': 'DOS',
    'DoS GoldenEye': 'DOS',
    'DoS Hulk': 'DOS',
    'DoS Slowhttptest': 'DOS',
    'DoS slowloris': 'DOS',
    'FTP-Patator': 'BRUTE_FORCE',
    'SSH-Patator': 'BRUTE_FORCE',
    'Heartbleed': 'WEB_ATTACK',
    'Infiltration': 'WEB_ATTACK',
    'PortScan': 'RECONNAISSANCE',
    'Web Attack � Brute Force': 'WEB_ATTACK',
    'Web Attack � Sql Injection': 'WEB_ATTACK',
    'Web Attack � XSS': 'WEB_ATTACK'
}
    
    # Apply the mapping
    df['Attack Type'] = df[column].map(attack_map)
    
    # Print the distribution of attack types
    print("Attack type distribution:")
    print(df['Attack Type'].value_counts())
    
    return df

def select_features(X, y, n_features=22):
    """
    Select the most important features using a trained Random Forest model.
    
    Args:
        X: Feature matrix
        y: Target variable
        n_features: Number of top features to select
        
    Returns:
        list of selected feature names
    """
    print(f"Selecting top {n_features} features...")
    
    # Train a basic Random Forest model to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for feature importance and sort it
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Select top n_features
    selected_features = feature_importance['feature'][:n_features].tolist()
    
    # Print the selected features
    print("Selected features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    
    return selected_features

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train and tune a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        The best Random Forest model
    """
    print("Training Random Forest model...")
    
    # Basic Random Forest (initial)
    start_fit = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    end_fit = time.time()
    
    print(f"Initial model training time: {end_fit - start_fit:.2f} seconds")
    
    # Predict and evaluate
    start_pred = time.time()
    y_pred = rf.predict(X_test)
    end_pred = time.time()
    
    print(f"Prediction time: {end_pred - start_pred:.2f} seconds")
    print("Initial model evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Hyperparameter tuning
    print("Optimizing Random Forest with hyperparameter tuning...")
    
    # Define parameters for grid search
    params = {
        'max_depth': list(range(12, 20)) + [None],
        'n_estimators': [50, 100, 150, 200],
        'class_weight': [None, 'balanced']
    }
    
    # Use HalvingGridSearchCV for efficient parameter search
    rf_grid = HalvingGridSearchCV(
        rf,
        params,
        cv=5,
        scoring='f1_macro',
        verbose=1,
        return_train_score=True,
        n_jobs=-1
    )
    
    start_fit = time.time()
    rf_grid.fit(X_train, y_train)
    end_fit = time.time()
    
    print(f"Grid search time: {end_fit - start_fit:.2f} seconds")
    print(f"Best parameters: {rf_grid.best_params_}")
    
    # Get the best model
    rf_best = rf_grid.best_estimator_
    
    # Evaluate the optimized model
    start_pred = time.time()
    y_pred = rf_best.predict(X_test)
    end_pred = time.time()
    
    print(f"Optimized model prediction time: {end_pred - start_pred:.2f} seconds")
    print("Optimized model evaluation:")
    print(classification_report(y_test, y_pred))
    
    return rf_best

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train and tune an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        The best XGBoost model
    """
    print("Training XGBoost model...")
    
    # Encode the labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Basic XGBoost model
    start_fit = time.time()
    xgb = XGBClassifier(n_estimators=50, random_state=42)
    xgb.fit(X_train, y_train_encoded)
    end_fit = time.time()
    
    print(f"Initial model training time: {end_fit - start_fit:.2f} seconds")
    
    # Predict and evaluate
    start_pred = time.time()
    y_pred_xgb = xgb.predict(X_test)
    end_pred = time.time()
    
    y_pred_decoded = le.inverse_transform(y_pred_xgb)
    
    print(f"Prediction time: {end_pred - start_pred:.2f} seconds")
    print("Initial model evaluation:")
    print(classification_report(y_test, y_pred_decoded))
    
    # Hyperparameter tuning
    print("Optimizing XGBoost with hyperparameter tuning...")
    
    # Define parameters for grid search
    params = {
        'max_depth': list(range(6, 13)) + [None],
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.5, 0.2, 0.05],
    }
    
    # Use HalvingGridSearchCV for efficient parameter search
    xgb_grid = HalvingGridSearchCV(
        estimator=xgb,
        param_grid=params,
        cv=5,
        scoring='f1_macro',
        verbose=1,
        return_train_score=True,
        n_jobs=-1
    )
    
    start_fit = time.time()
    xgb_grid.fit(X_train, y_train_encoded)
    end_fit = time.time()
    
    print(f"Grid search time: {end_fit - start_fit:.2f} seconds")
    print(f"Best parameters: {xgb_grid.best_params_}")
    
    # Get the best model
    xgb_best = xgb_grid.best_estimator_
    
    # Store the label encoder with the model for later use
    xgb_best.label_encoder_ = le
    
    # Evaluate the optimized model
    start_pred = time.time()
    y_pred_xgb = xgb_best.predict(X_test)
    end_pred = time.time()
    
    y_pred_decoded = le.inverse_transform(y_pred_xgb)
    
    print(f"Optimized model prediction time: {end_pred - start_pred:.2f} seconds")
    print("Optimized model evaluation:")
    print(classification_report(y_test, y_pred_decoded))
    
    return xgb_best

def compare_models(rf_model, xgb_model, X_test, y_test):
    """
    Compare Random Forest and XGBoost models and select the best one.
    
    Args:
        rf_model: Random Forest model
        xgb_model: XGBoost model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        The best model (either rf_model or xgb_model) and a dict of evaluation metrics
    """
    print("Comparing models...")
    
    # Predictions
    rf_pred = rf_model.predict(X_test)
    
    # For XGBoost, we need to convert the predictions back to original labels
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred = xgb_model.label_encoder_.inverse_transform(xgb_pred)
    
    # Get classification reports
    rf_report = classification_report(y_test, rf_pred, output_dict=True)
    xgb_report = classification_report(y_test, xgb_pred, output_dict=True)
    
    # Get macro average F1 scores
    rf_f1 = rf_report['macro avg']['f1-score']
    xgb_f1 = xgb_report['macro avg']['f1-score']
    
    print(f"Random Forest macro avg F1: {rf_f1:.4f}")
    print(f"XGBoost macro avg F1: {xgb_f1:.4f}")
    
    # Check false negatives for critical categories
    attack_types = ['WEB_ATTACK', 'BOTNET']
    rf_critical_fn = 0
    xgb_critical_fn = 0
    
    for attack_type in attack_types:
        # For Random Forest
        true_indices = (y_test == attack_type)
        if true_indices.any():
            rf_fn = ((rf_pred[true_indices] != attack_type).sum() / true_indices.sum())
            rf_critical_fn += rf_fn
            print(f"RF False Negative Rate for {attack_type}: {rf_fn:.4f}")
        
        # For XGBoost
        xgb_fn = ((xgb_pred[true_indices] != attack_type).sum() / true_indices.sum()) if true_indices.any() else 0
        xgb_critical_fn += xgb_fn
        print(f"XGB False Negative Rate for {attack_type}: {xgb_fn:.4f}")
    
    # Select the best model based on a combination of macro F1 and critical FN rates
    # Giving more weight to avoiding false negatives in critical categories
    rf_score = rf_f1 - (0.2 * rf_critical_fn)
    xgb_score = xgb_f1 - (0.2 * xgb_critical_fn)
    
    print(f"Random Forest adjusted score: {rf_score:.4f}")
    print(f"XGBoost adjusted score: {xgb_score:.4f}")
    
    if rf_score >= xgb_score:
        print("Random Forest is the better model.")
        best_model = rf_model
        model_type = 'random_forest'
    else:
        print("XGBoost is the better model.")
        best_model = xgb_model
        model_type = 'xgboost'
    
    # Return the best model and evaluation metrics
    metrics = {
        'rf_f1': rf_f1,
        'xgb_f1': xgb_f1,
        'rf_critical_fn': rf_critical_fn,
        'xgb_critical_fn': xgb_critical_fn,
        'best_model_type': model_type
    }
    
    return best_model, metrics

def save_model(model, model_metrics, feature_importance, selected_features, model_dir):
    """
    Save the trained model and related information.
    
    Args:
        model: The trained model to save
        model_metrics: Dictionary of evaluation metrics
        feature_importance: DataFrame of feature importance
        selected_features: List of selected feature names
        model_dir: Directory to save the model files
    """
    print(f"Saving model to {model_dir}...")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create metadata
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_type': model_metrics['best_model_type'],
        'evaluation': {
            'rf_f1': model_metrics['rf_f1'],
            'xgb_f1': model_metrics['xgb_f1'],
            'rf_critical_fn': model_metrics['rf_critical_fn'],
            'xgb_critical_fn': model_metrics['xgb_critical_fn']
        },
        'selected_features': selected_features
    }
    
    # Save model
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)
    
    # Save metadata
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Save selected features
    with open(os.path.join(model_dir, 'selected_features.json'), 'w') as f:
        json.dump(selected_features, f, indent=4)
    
    print("Model and related files saved successfully.")

def visualize_results(model, X_test, y_test, selected_features, model_dir):
    """
    Generate and save visualizations of the model results.
    
    Args:
        model: The trained model
        X_test: Test features
        y_test: Test labels
        selected_features: List of selected feature names
        model_dir: Directory to save the visualizations
    """
    print("Generating visualizations...")
    
    # Create visualizations directory
    viz_dir = os.path.join(model_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get predictions
    if hasattr(model, 'label_encoder_'):
        # XGBoost model
        y_pred = model.predict(X_test)
        y_pred = model.label_encoder_.inverse_transform(y_pred)
    else:
        # Random Forest model
        y_pred = model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_test.columns
        
        # Create a DataFrame for feature importance and sort it
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(selected_features)), 
                feature_importance.iloc[:len(selected_features)]['importance'], 
                align='center')
        plt.yticks(range(len(selected_features)), 
                  feature_importance.iloc[:len(selected_features)]['feature'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {viz_dir}")

def read_eve_json(eve_json_path):
    """
    Read and parse Suricata Eve JSON log file.
    
    Args:
        eve_json_path: Path to the eve.json file
        
    Returns:
        DataFrame containing parsed Eve JSON records
    """
    print(f"Reading Eve JSON file: {eve_json_path}")
    
    # Read the file line by line as each line is a separate JSON object
    data = []
    with open(eve_json_path, 'r') as f:
        for line in f:
            try:
                # Parse the JSON object
                record = json.loads(line)
                
                # Check if this is the new format (with 'suricata.eve' nesting)
                if 'suricata' in record and 'eve' in record['suricata']:
                    event_data = record['suricata']['eve']
                    data.append(event_data)
                else:
                    # Try the old format as fallback
                    if 'event_type' in record:
                        data.append(record)
                    
            except json.JSONDecodeError:
                print(f"Error parsing JSON line: {line[:50]}...")
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} records from Eve JSON file")
    
    return df

def extract_features_from_eve(df):
    """
    Extract and transform features from Eve JSON records to match model features.
    
    Args:
        df: DataFrame with parsed Eve JSON records
        
    Returns:
        DataFrame with extracted features matching the model's expected features
    """
    print("Extracting features from Eve JSON records...")
    
    # Filter out records with event_type = "alert"
    df = df[df['event_type'] != 'alert']
    print(f"After filtering alerts: {len(df)} records")
    
    # Initialize an empty DataFrame to store the extracted features
    features = pd.DataFrame()
    
    # Map common fields
    features['Flow ID'] = df.get('flow_id', None)
    features['Source IP'] = df.get('src_ip', None)
    features['Source Port'] = df.get('src_port', None)
    features['Destination IP'] = df.get('dest_ip', None)
    features['Destination Port'] = df.get('dest_port', None)
    features['Protocol'] = df.get('proto', None)
    
    # Extract flow information
    for idx, row in df.iterrows():
        # Check if 'flow' is in the record
        if 'flow' in row:
            flow = row['flow']
            features.at[idx, 'Flow Duration'] = flow.get('age', 0) if isinstance(flow, dict) else 0
            features.at[idx, 'Total Fwd Packets'] = flow.get('pkts_toserver', 0) if isinstance(flow, dict) else 0
            features.at[idx, 'Total Backward Packets'] = flow.get('pkts_toclient', 0) if isinstance(flow, dict) else 0
            features.at[idx, 'Total Length of Fwd Packets'] = flow.get('bytes_toserver', 0) if isinstance(flow, dict) else 0
            features.at[idx, 'Total Length of Bwd Packets'] = flow.get('bytes_toclient', 0) if isinstance(flow, dict) else 0
        else:
            features.at[idx, 'Flow Duration'] = 0
            features.at[idx, 'Total Fwd Packets'] = 0
            features.at[idx, 'Total Backward Packets'] = 0
            features.at[idx, 'Total Length of Fwd Packets'] = 0
            features.at[idx, 'Total Length of Bwd Packets'] = 0
        
        # Calculate derived metrics
        features.at[idx, 'Flow Bytes/s'] = features.at[idx, 'Total Length of Fwd Packets'] + features.at[idx, 'Total Length of Bwd Packets']
        features.at[idx, 'Flow Packets/s'] = features.at[idx, 'Total Fwd Packets'] + features.at[idx, 'Total Backward Packets']
        
        # Add more features based on available data in eve.json
        # This is a starting point - you would need to map all required features
    
    # Fill any missing values
    features.fillna(0, inplace=True)
    
    return features

def create_baseline(features):
    """
    Create a statistical baseline profile from normal network traffic.
    
    Args:
        features: DataFrame with extracted features
        
    Returns:
        Dictionary containing baseline statistics
    """
    print("Creating baseline profile from normal traffic...")
    
    # Calculate statistical measures for each feature
    baseline = {}
    
    for column in features.columns:
        if column in ['Flow ID', 'Source IP', 'Destination IP', 'Protocol']:
            # Skip non-numeric columns
            continue
        
        # Calculate statistics
        column_stats = {
            'mean': features[column].mean(),
            'std': features[column].std(),
            'min': features[column].min(),
            'max': features[column].max(),
            'q1': features[column].quantile(0.25),
            'median': features[column].median(),
            'q3': features[column].quantile(0.75),
            # Calculate IQR for outlier detection
            'iqr': features[column].quantile(0.75) - features[column].quantile(0.25)
        }
        
        baseline[column] = column_stats
    
    # Add metadata
    baseline['metadata'] = {
        'timestamp': datetime.datetime.now().isoformat(),
        'num_samples': len(features),
        'features': list(features.columns)
    }
    
    return baseline

def save_baseline(baseline, model_dir):
    """
    Save the baseline profile.
    
    Args:
        baseline: Dictionary containing baseline statistics
        model_dir: Directory to save the baseline
    """
    print(f"Saving baseline profile to {model_dir}...")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save baseline
    with open(os.path.join(model_dir, 'baseline.json'), 'w') as f:
        json.dump(baseline, f, indent=4)
    
    print("Baseline profile saved successfully.")

def train_mode(args):
    """
    Implement the training mode to train classification models on CICIDS2017 dataset.
    
    Args:
        args: Command-line arguments
    """
    # Load dataset
    df = load_cicids2017(args.dataset)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Map attack categories
    df = map_attack_categories(df)
    
    # Split features and target
    X = df.drop(columns=['Attack Type', 'Label'])
    y = df['Attack Type']
    
    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Feature selection
    selected_features = select_features(X_train, y_train, n_features=22)
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Train models
    rf_model = train_random_forest(X_train_selected, y_train, X_test_selected, y_test)
    xgb_model = train_xgboost(X_train_selected, y_train, X_test_selected, y_test)
    
    # Compare and select the best model
    best_model, model_metrics = compare_models(rf_model, xgb_model, X_test_selected, y_test)
    
    # Get feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
    else:
        # XGBoost model
        importances = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    # Save the model and related information
    save_model(best_model, model_metrics, feature_importance, selected_features, args.model)
    
    # Generate and save visualizations
    visualize_results(best_model, X_test_selected, y_test, selected_features, args.model)

def baseline_mode(args):
    """
    Implement the baseline mode to learn normal network behavior from eve.json files.
    
    Args:
        args: Command-line arguments
    """
    # Read Eve JSON file
    df = read_eve_json(args.evejson)
    
    # Extract features from Eve JSON records
    features = extract_features_from_eve(df)
    
    # Create baseline profile
    baseline = create_baseline(features)
    
    # Save baseline profile
    save_baseline(baseline, args.model)

def main():
    """
    Main function to parse arguments and execute the selected mode.
    """
    parser = argparse.ArgumentParser(description='Network Intrusion Detection System - Training Module')
    parser.add_argument('--mode', choices=['train', 'baseline'], required=True, 
                        help='Operating mode: train (for model training) or baseline (for baseline creation)')
    parser.add_argument('--dataset', help='Path to CICIDS2017 dataset')
    parser.add_argument('--model', required=True, help='Path to model output directory')
    parser.add_argument('--evejson', help='Path to eve.json file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'train' and not args.dataset:
        parser.error("--dataset is required when using --mode train")
    
    if args.mode == 'baseline' and not args.evejson:
        parser.error("--evejson is required when using --mode baseline")
    
    # Execute the selected mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'baseline':
        baseline_mode(args)
    else:
        parser.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()