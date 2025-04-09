#!/usr/bin/env python3
"""
Hybrid NIDPS - Training Module

This script trains machine learning models on the CICIDS2017 dataset.

Usage:
    python train.py --dataset <path_to_dataset> --model_dir <output_directory>
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import datetime
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def parse_args():
    """Parse command-line arguments with environment variable fallbacks."""
    parser = argparse.ArgumentParser(description='Hybrid NIDPS - Training Module')
    parser.add_argument('--dataset', 
                        default=os.getenv('DATASET_PATH'),
                        help='Path to CICIDS2017 dataset')
    parser.add_argument('--model_dir', 
                        default=os.getenv('MODEL_DIR', './model'),
                        help='Directory to save trained models')
    
    args = parser.parse_args()
    
    # Validate required parameters
    if not args.dataset:
        parser.error("Dataset path is required. Provide it via --dataset or set DATASET_PATH in .env")
    
    return args

def load_cicids2017(dataset_path):
    """Load and preprocess the CICIDS2017 dataset."""
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
    """Preprocess the dataset."""
    print("Preprocessing data...")
    original_shape = df.shape
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Remove rows with NaN values
    df.dropna(inplace=True)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Report data reduction
    print(f"Original shape: {original_shape}")
    print(f"After preprocessing: {df.shape}")
    
    return df

def select_features(X, y, n_features=22):
    """Select the most important features using a trained Random Forest model."""
    print(f"Selecting top {n_features} features...")
    
    # Train a basic Random Forest model to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
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
    
    return selected_features, feature_importance

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train a Decision Tree classifier with regularization to prevent overfitting."""
    print("Training Decision Tree model...")
    
    # Use cross-validation to choose the best max_depth
    print("Finding optimal max_depth with cross-validation...")
    best_depth = 0
    best_score = 0
    depth_range = range(5, 25, 2)
    
    for depth in depth_range:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_val_score(dt, X_train, y_train, cv=5, scoring='f1_macro')
        mean_score = scores.mean()
        print(f"Max depth: {depth}, Cross-validation score: {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_depth = depth
    
    print(f"Best max_depth: {best_depth} with CV score: {best_score:.4f}")
    
    # Train final model with best depth
    dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    dt.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = dt.predict(X_test)
    print("Decision Tree evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Additional evaluation to detect overfitting
    train_score = dt.score(X_train, y_train)
    test_score = dt.score(X_test, y_test)
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    print(f"Difference (potential overfitting indicator): {train_score - test_score:.4f}")
    
    return dt

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train a Random Forest classifier with regularization to prevent overfitting."""
    print("Training Random Forest model...")
    
    # Train model with regularization parameters
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_macro')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Train final model
    rf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf.predict(X_test)
    print("Random Forest evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Additional evaluation to detect overfitting
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    print(f"Difference (potential overfitting indicator): {train_score - test_score:.4f}")
    
    return rf

def save_models(dt_model, rf_model, feature_importance, selected_features, model_dir, X_train, y_train):
    """Save the trained models and related information."""
    print(f"Saving models to {model_dir}...")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save models
    with open(os.path.join(model_dir, 'dt_model.pkl'), 'wb') as f:
        pickle.dump(dt_model, f)
    
    with open(os.path.join(model_dir, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Save selected features
    with open(os.path.join(model_dir, 'selected_features.json'), 'w') as f:
        json.dump(selected_features, f)
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)
    
    # Create and save statistical baseline of normal traffic
    print("Creating statistical baseline of normal traffic...")
    X_normal = X_train[y_train == 0]  # Only normal traffic (non-attack)
    X_normal_selected = X_normal[selected_features]
    
    # Calculate statistical measures for each feature
    baseline = {}
    for feature in selected_features:
        if feature in X_normal_selected.columns:
            baseline[feature] = {
                'mean': float(X_normal_selected[feature].mean()),
                'std': float(X_normal_selected[feature].std()),
                'min': float(X_normal_selected[feature].min()),
                'max': float(X_normal_selected[feature].max()),
                'q1': float(X_normal_selected[feature].quantile(0.25)),
                'median': float(X_normal_selected[feature].median()),
                'q3': float(X_normal_selected[feature].quantile(0.75)),
                'iqr': float(X_normal_selected[feature].quantile(0.75) - X_normal_selected[feature].quantile(0.25))
            }
    
    # Save baseline
    with open(os.path.join(model_dir, 'baseline.json'), 'w') as f:
        json.dump(baseline, f, indent=4)
    
    print("Models and related files saved successfully.")

def main():
    """Main function."""
    args = parse_args()
    
    # Load dataset
    df = load_cicids2017(args.dataset)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Extract features and target
    X = df.drop('Label', axis=1) if 'Label' in df.columns else df.drop('label', axis=1)
    
    # For CICIDS2017, convert labels to binary (normal=0, attack=1)
    y = df['Label'] if 'Label' in df.columns else df['label']
    y = np.where((y == 'BENIGN') | (y == 'benign'), 0, 1)
    
    # Split data for training and testing - use temporal split if possible
    # to better simulate real-world conditions
    if 'timestamp' in df.columns:
        # Sort by timestamp and split
        df_sorted = df.sort_values('timestamp')
        train_size = int(len(df_sorted) * 0.75)
        X_train = df_sorted.iloc[:train_size].drop(['Label', 'timestamp'], axis=1)
        y_train = np.where((df_sorted.iloc[:train_size]['Label'] == 'BENIGN'), 0, 1)
        X_test = df_sorted.iloc[train_size:].drop(['Label', 'timestamp'], axis=1)
        y_test = np.where((df_sorted.iloc[train_size:]['Label'] == 'BENIGN'), 0, 1)
    else:
        # Use stratified split if no timestamp is available
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    
    # Feature selection
    selected_features, feature_importance = select_features(X_train, y_train)
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Scale features to improve model performance and prevent overfitting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Convert back to DataFrame for feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_selected.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_selected.columns)
    
    # Train models
    dt_model = train_decision_tree(X_train_scaled_df, y_train, X_test_scaled_df, y_test)
    rf_model = train_random_forest(X_train_scaled_df, y_train, X_test_scaled_df, y_test)
    
    # Save models
    save_models(dt_model, rf_model, feature_importance, selected_features, args.model_dir, 
                X_train_scaled_df, y_train)
    
    print("Training completed successfully.")

if __name__ == "__main__":
    main()