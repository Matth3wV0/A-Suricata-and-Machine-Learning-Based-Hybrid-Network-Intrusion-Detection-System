#!/usr/bin/env python3
"""
NIDS Training Module

This script processes the CICIDS2017 dataset and trains a multi-tiered hybrid intrusion
detection system with both classification and anomaly detection capabilities.

Usage:
    python nids_train.py --dataset DATASET_PATH --model MODEL_OUTPUT_DIR [--baseline EVE_JSON_PATH]

Arguments:
    --dataset      Path to CICIDS2017 dataset (.csv files)
    --model        Path to model output directory
    --baseline     (Optional) Path to eve.json file to create baseline profile

Example:
    python nids_train.py --dataset CICIDS2017/ --model models/
    python nids_train.py --dataset CICIDS2017/ --model models/ --baseline eve.json
"""

import os
import sys
import time
import json
import pickle
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import MiniBatchKMeans
from imblearn.over_sampling import SMOTE
import xgboost as xgb

try:
    # Optional libraries - will use if available
    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    print("Warning: hyperopt not available. Will use default hyperparameters.")

# Set default parameters
N_FEATURES = 22  # Number of important features to select
FEATURE_IMPORTANCE_THRESHOLD = 0.9  # Threshold for cumulative feature importance
N_CLUSTERS = 16  # Default number of clusters for anomaly detection


class FeatureSelector:
    """
    Feature selection using Information Gain and correlation-based methods
    """
    def __init__(self, n_features=N_FEATURES):
        """Initialize the feature selector"""
        self.n_features = n_features
        self.selected_features = None
        self.feature_ranks = None

    def select_features(self, X, y):
        """
        Select the most important features based on Information Gain
        
        Args:
            X: Features DataFrame
            y: Target variable
            
        Returns:
            List of selected feature names
        """
        print(f"Selecting top {self.n_features} features using Information Gain...")
        
        # Calculate information gain for each feature
        feature_names = X.columns
        importances = mutual_info_classif(X, y)
        
        # Create sorted list of features by importance
        self.feature_ranks = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Get cumulative importance
        self.feature_ranks['cumulative_importance'] = self.feature_ranks['importance'].cumsum() / self.feature_ranks['importance'].sum()
        
        # Safety check: handle empty or all-zero importance
        if self.feature_ranks.empty or self.feature_ranks['importance'].sum() == 0:
            print("Warning: No informative features found. Falling back to first N features.")
            selected_features = list(feature_names[:self.n_features])
        else:
            selected_index = self.feature_ranks['cumulative_importance'].searchsorted(FEATURE_IMPORTANCE_THRESHOLD)
            # Ensure selected_index is an integer
            selected_index = int(selected_index) if np.isscalar(selected_index) else selected_index[0]
            selected_features = list(self.feature_ranks['feature'][:max(selected_index + 1, self.n_features)])

        
        print(f"Selected {len(selected_features)} features with cumulative importance of {FEATURE_IMPORTANCE_THRESHOLD*100:.1f}%")
        
        # Print top features and their importance
        print("\nTop 10 most important features:")
        for i, (_, row) in enumerate(self.feature_ranks.iloc[:10].iterrows(), 1):
            print(f"{i}. {row['feature']} - {row['importance']:.4f}")
        
        self.selected_features = selected_features
        return selected_features

    def filter_correlated_features(self, X, threshold=0.9):
        """
        Filter out highly correlated features
        
        Args:
            X: Features DataFrame
            threshold: Correlation threshold to filter features
            
        Returns:
            List of feature names after removing correlated features
        """
        if self.selected_features is None:
            raise ValueError("Must run select_features before filter_correlated_features")
        
        # Calculate correlation matrix
        X_selected = X[self.selected_features]
        corr_matrix = X_selected.corr().abs()
        
        # Create upper triangle mask
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            print(f"\nDropping {len(to_drop)} correlated features:")
            for feature in to_drop:
                print(f"- {feature}")
            
            # Remove correlated features from selected features
            final_features = [feature for feature in self.selected_features if feature not in to_drop]
            self.selected_features = final_features
            
            print(f"Retained {len(final_features)} features after removing correlated features")
        else:
            print("\nNo highly correlated features found")
        
        return self.selected_features
    
    def plot_feature_importance(self, output_path=None):
        """
        Plot the importance of selected features
        
        Args:
            output_path: Path to save the plot (optional)
        """
        if self.feature_ranks is None:
            raise ValueError("Must run select_features before plotting feature importance")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        
        # Use only selected features for the plot
        plot_data = self.feature_ranks[self.feature_ranks['feature'].isin(self.selected_features)]
        
        # Create bar plot
        sns.barplot(x='importance', y='feature', data=plot_data, palette='viridis')
        plt.title(f'Top {len(self.selected_features)} Features by Information Gain')
        plt.xlabel('Information Gain')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {output_path}")
        
        plt.close()
        

class MultiTieredHybridIDS:
    """
    Implementation of a Multi-Tiered Hybrid Intrusion Detection System (MTH-IDS)
    with both signature-based and anomaly-based detection capabilities.
    """
    
    def __init__(self):
        """Initialize the MTH-IDS model components"""
        # Base classifier models
        self.dt_model = None
        self.rf_model = None
        self.et_model = None
        self.xgb_model = None
        
        # Ensemble model
        self.stacking_model = None
        
        # Anomaly detection model
        self.kmeans_model = None
        self.n_clusters = N_CLUSTERS
        self.cluster_to_class_mapping = None
        
        # Biased classifiers for handling false positives and negatives
        self.biased_classifier_fp = None
        self.biased_classifier_fn = None
        
        # Feature engineering components
        self.feature_selector = FeatureSelector()
        self.scaler = StandardScaler()
        
        # Other utilities
        self.label_encoder = None
        self.baseline_profile = None
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset by handling missing and infinite values
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        print("\n=== Data Preprocessing ===")
        original_shape = df.shape
        
        # Check for infinite values
        is_inf = df.isin([np.inf, -np.inf])
        inf_counts = is_inf.sum()
        columns_with_inf = inf_counts[inf_counts > 0]
        
        if not columns_with_inf.empty:
            print(f"Found {columns_with_inf.sum()} infinite values in {len(columns_with_inf)} columns")
            # Replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        columns_with_nan = nan_counts[nan_counts > 0]
        
        if not columns_with_nan.empty:
            print(f"Found {nan_counts.sum()} NaN values in {len(columns_with_nan)} columns")
            
            # Fill NaN values with median for each column
            for col in columns_with_nan.index:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Check and remove duplicate rows
        duplicate_rows_count = df.duplicated().sum()
        if duplicate_rows_count > 0:
            print(f"Found {duplicate_rows_count} duplicate rows")
            df.drop_duplicates(inplace=True)
        
        # Report data reduction
        print(f"Original shape: {original_shape}")
        print(f"After preprocessing: {df.shape}")
        print(f"Reduction: {original_shape[0] - df.shape[0]} rows ({(original_shape[0] - df.shape[0])/original_shape[0]*100:.2f}%)")
        
        return df
    
    def map_attack_categories(self, df, attack_field='Label'):
        """
        Map attack types to broader categories
        
        Args:
            df: DataFrame containing attack labels
            attack_field: Name of the field containing attack labels
            
        Returns:
            DataFrame with added 'Attack Type' column
        """
        print("\n=== Mapping Attack Categories ===")
        
        # Define attack category mapping
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
        df['Attack Type'] = df[attack_field].map(attack_map)
        
        # Print the distribution of attack types
        attack_counts = df['Attack Type'].value_counts()
        print("Attack type distribution:")
        for attack_type, count in attack_counts.items():
            print(f"  {attack_type}: {count} ({count/len(df)*100:.2f}%)")
        
        # Create binary label (0 for normal, 1 for attack)
        df['Binary Label'] = df['Attack Type'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        return df
    
    def train(self, df, test_size=0.25, balance_classes=True, hyperopt=True):
        """
        Train the complete MTH-IDS model
        
        Args:
            df: Input DataFrame with 'Attack Type' column
            test_size: Proportion of data to use for testing
            balance_classes: Whether to balance classes using SMOTE
            hyperopt: Whether to use hyperopt for hyperparameter optimization
            
        Returns:
            Self (trained model)
        """
        print("\n=== Training MTH-IDS Model ===")
        
        # Split features and target
        X = df.drop(columns=['Attack Type', 'Label', 'Binary Label'], errors='ignore')
        y_multi = df['Attack Type']  # For multi-class classification
        y_binary = df['Binary Label']  # For binary classification
        
        # Z-score normalization
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        # Select important features
        selected_features = self.feature_selector.select_features(X_scaled, y_multi)
        
        # Filter correlated features
        final_features = self.feature_selector.filter_correlated_features(X_scaled)
        
        # Use selected features
        X_selected = X_scaled[final_features]
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_multi, test_size=test_size, random_state=42, stratify=y_multi
        )
        
        # Balance classes if needed
        if balance_classes:
            print("\n=== Balancing Classes with SMOTE ===")
            # Count classes
            class_counts = pd.Series(y_train).value_counts()
            print("Original class distribution:")
            print(class_counts)
            
            # Set up SMOTE
            minority_classes = class_counts[class_counts < 1000].index.tolist()
            if minority_classes:
                # Define SMOTE strategy
                sampling_strategy = {cls: max(1000, int(class_counts.median())) 
                                    for cls in minority_classes}
                
                # Apply SMOTE
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                
                # Print new distribution
                new_counts = pd.Series(y_train).value_counts()
                print("Balanced class distribution:")
                print(new_counts)
            else:
                print("No minority classes found, skipping SMOTE")
        
        # Train base classifiers
        print("\n=== Training Base Classifiers (Tier 1) ===")
        
        # Decision Tree
        print("Training Decision Tree...")
        self.dt_model = DecisionTreeClassifier(random_state=42)
        self.dt_model.fit(X_train, y_train)
        
        # Random Forest
        print("Training Random Forest...")
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
        # Extra Trees
        print("Training Extra Trees...")
        self.et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
        self.et_model.fit(X_train, y_train)
        
        # XGBoost
        print("Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        self.xgb_model.fit(X_train, y_train)
        
        # Optimize hyperparameters if requested
        if hyperopt and HYPEROPT_AVAILABLE:
            print("\n=== Optimizing Hyperparameters (Tier 2) ===")
            self._optimize_hyperparameters(X_train, y_train)
            
            # Retrain models with optimized parameters
            self.dt_model.fit(X_train, y_train)
            self.rf_model.fit(X_train, y_train)
            self.et_model.fit(X_train, y_train)
            self.xgb_model.fit(X_train, y_train)
        
        # Train stacking ensemble
        print("\n=== Training Stacking Ensemble ===")
        self._train_stacking_ensemble(X_train, y_train)
        
        # Train anomaly detection
        print("\n=== Training Anomaly Detection (Tier 3) ===")
        self._train_anomaly_detection(X_train, y_train, optimize=hyperopt)
        
        # Evaluate on test set
        print("\n=== Evaluating Models ===")
        self._evaluate_models(X_test, y_test)
        
        return self
    
    def _optimize_hyperparameters(self, X_train, y_train):
        """
        Optimize hyperparameters for base classifiers using hyperopt
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        if not HYPEROPT_AVAILABLE:
            print("Hyperopt not available, skipping hyperparameter optimization")
            return
        
        # Define objective function for Decision Tree
        def dt_objective(params):
            params = {
                'max_depth': int(params['max_depth']),
                'min_samples_split': int(params['min_samples_split']),
                'min_samples_leaf': int(params['min_samples_leaf']),
                'criterion': params['criterion']
            }
            
            # Train model with these parameters
            model = DecisionTreeClassifier(random_state=42, **params)
            model.fit(X_train, y_train)
            
            # Use F1 score as the metric
            y_pred = model.predict(X_train)
            f1 = f1_score(y_train, y_pred, average='weighted')
            
            return {'loss': -f1, 'status': STATUS_OK}
        
        # Define search space for Decision Tree
        dt_space = {
            'max_depth': hp.quniform('dt_max_depth', 5, 30, 1),
            'min_samples_split': hp.quniform('dt_min_samples_split', 2, 20, 1),
            'min_samples_leaf': hp.quniform('dt_min_samples_leaf', 1, 10, 1),
            'criterion': hp.choice('dt_criterion', ['gini', 'entropy'])
        }
        
        # Optimize Decision Tree
        print("Optimizing Decision Tree...")
        dt_best = fmin(fn=dt_objective, space=dt_space, algo=tpe.suggest, max_evals=10)
        
        # Update Decision Tree model with best parameters
        criterion = ['gini', 'entropy'][dt_best.get('dt_criterion', 0)]
        self.dt_model = DecisionTreeClassifier(
            max_depth=int(dt_best.get('dt_max_depth', 10)),
            min_samples_split=int(dt_best.get('dt_min_samples_split', 2)),
            min_samples_leaf=int(dt_best.get('dt_min_samples_leaf', 1)),
            criterion=criterion,
            random_state=42
        )
        
        # Define objective function for Random Forest
        def rf_objective(params):
            params = {
                'n_estimators': int(params['n_estimators']),
                'max_depth': int(params['max_depth']),
                'min_samples_split': int(params['min_samples_split']),
                'min_samples_leaf': int(params['min_samples_leaf'])
            }
            
            # Train model with these parameters
            model = RandomForestClassifier(random_state=42, **params)
            model.fit(X_train, y_train)
            
            # Use F1 score as the metric
            y_pred = model.predict(X_train)
            f1 = f1_score(y_train, y_pred, average='weighted')
            
            return {'loss': -f1, 'status': STATUS_OK}
        
        # Define search space for Random Forest
        rf_space = {
            'n_estimators': hp.quniform('rf_n_estimators', 50, 200, 10),
            'max_depth': hp.quniform('rf_max_depth', 10, 50, 5),
            'min_samples_split': hp.quniform('rf_min_samples_split', 2, 10, 1),
            'min_samples_leaf': hp.quniform('rf_min_samples_leaf', 1, 5, 1)
        }
        
        # Optimize Random Forest
        print("Optimizing Random Forest...")
        rf_best = fmin(fn=rf_objective, space=rf_space, algo=tpe.suggest, max_evals=10)
        
        # Update Random Forest model with best parameters
        self.rf_model = RandomForestClassifier(
            n_estimators=int(rf_best.get('rf_n_estimators', 100)),
            max_depth=int(rf_best.get('rf_max_depth', 20)),
            min_samples_split=int(rf_best.get('rf_min_samples_split', 2)),
            min_samples_leaf=int(rf_best.get('rf_min_samples_leaf', 1)),
            random_state=42
        )
        
        # Define objective function for XGBoost
        def xgb_objective(params):
            params = {
                'n_estimators': int(params['n_estimators']),
                'max_depth': int(params['max_depth']),
                'learning_rate': params['learning_rate'],
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree']
            }
            
            # Train model with these parameters
            model = xgb.XGBClassifier(random_state=42, **params)
            model.fit(X_train, y_train)
            
            # Use F1 score as the metric
            y_pred = model.predict(X_train)
            f1 = f1_score(y_train, y_pred, average='weighted')
            
            return {'loss': -f1, 'status': STATUS_OK}
        
        # Define search space for XGBoost
        xgb_space = {
            'n_estimators': hp.quniform('xgb_n_estimators', 50, 200, 10),
            'max_depth': hp.quniform('xgb_max_depth', 3, 10, 1),
            'learning_rate': hp.loguniform('xgb_learning_rate', -3, 0),  # 0.05 to 1.0
            'subsample': hp.uniform('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.6, 1.0)
        }
        
        # Optimize XGBoost
        print("Optimizing XGBoost...")
        xgb_best = fmin(fn=xgb_objective, space=xgb_space, algo=tpe.suggest, max_evals=10)
        
        # Update XGBoost model with best parameters
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=int(xgb_best.get('xgb_n_estimators', 100)),
            max_depth=int(xgb_best.get('xgb_max_depth', 6)),
            learning_rate=float(np.exp(xgb_best.get('xgb_learning_rate', -1))),
            subsample=float(xgb_best.get('xgb_subsample', 0.8)),
            colsample_bytree=float(xgb_best.get('xgb_colsample_bytree', 0.8)),
            random_state=42
        )
    
    def _train_stacking_ensemble(self, X_train, y_train):
        """
        Train a stacking ensemble model using the base classifiers
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Generate predictions from base models
        dt_preds = self.dt_model.predict(X_train).reshape(-1, 1)
        rf_preds = self.rf_model.predict(X_train).reshape(-1, 1)
        et_preds = self.et_model.predict(X_train).reshape(-1, 1)
        xgb_preds = self.xgb_model.predict(X_train).reshape(-1, 1)
        
        # Combine predictions into a new feature matrix
        meta_features = np.hstack((dt_preds, rf_preds, et_preds, xgb_preds))
        
        # Train meta-classifier (XGBoost)
        self.stacking_model = xgb.XGBClassifier(n_estimators=50, random_state=42)
        self.stacking_model.fit(meta_features, y_train)
        
        print("Stacking ensemble trained successfully")
    
    def _train_anomaly_detection(self, X_train, y_train, optimize=True):
        """
        Train the anomaly detection model using KMeans clustering
        
        Args:
            X_train: Training features
            y_train: Training labels
            optimize: Whether to optimize the number of clusters
        """
        # Optimize number of clusters if requested
        if optimize and HYPEROPT_AVAILABLE:
            print("Optimizing number of clusters for KMeans...")
            self._optimize_kmeans(X_train, y_train)
        
        # Train k-means with the optimal number of clusters
        print(f"Training KMeans with {self.n_clusters} clusters...")
        self.kmeans_model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=100, random_state=42)
        
        # Fit the model
        cluster_labels = self.kmeans_model.fit_predict(X_train)
        
        # Determine cluster-to-class mapping
        self.cluster_to_class_mapping = {}
        for cluster_id in range(self.n_clusters):
            # Get all samples in this cluster
            cluster_samples = y_train[cluster_labels == cluster_id]
            
            if len(cluster_samples) > 0:
                # Count occurrences of each class
                unique, counts = np.unique(cluster_samples, return_counts=True)
                
                # Assign cluster to the majority class
                majority_class = unique[np.argmax(counts)]
                self.cluster_to_class_mapping[cluster_id] = majority_class
            else:
                # If no samples in cluster, assign to class BENIGN
                self.cluster_to_class_mapping[cluster_id] = 'BENIGN'
        
        # Identify false positives and false negatives in the training set
        y_pred_kmeans = np.array([self.cluster_to_class_mapping.get(label, 'BENIGN') 
                                for label in cluster_labels])
        
        # Create binary predictions for calculating FP/FN
        y_train_binary = np.array([0 if cls == 'BENIGN' else 1 for cls in y_train])
        y_pred_binary = np.array([0 if cls == 'BENIGN' else 1 for cls in y_pred_kmeans])
        
        # Find false positives and false negatives
        fn_indices = np.where((y_train_binary == 1) & (y_pred_binary == 0))[0]
        fp_indices = np.where((y_train_binary == 0) & (y_pred_binary == 1))[0]
        
        # Train biased classifiers if we have enough samples
        if len(fn_indices) > 10:
            print(f"Training biased classifier for false negatives ({len(fn_indices)} samples)...")
            
            # Sample an equal number of normal instances
            normal_indices = np.where(y_train_binary == 0)[0]
            normal_sampled = np.random.choice(normal_indices, min(len(fn_indices), len(normal_indices)), replace=False)
            
            # Combine FN samples with normal samples
            X_fn = np.vstack((X_train[fn_indices], X_train[normal_sampled]))
            y_fn = np.concatenate((y_train_binary[fn_indices], y_train_binary[normal_sampled]))
            
            # Train the FN biased classifier
            self.biased_classifier_fn = RandomForestClassifier(n_estimators=50, random_state=42)
            self.biased_classifier_fn.fit(X_fn, y_fn)
        else:
            print("Not enough false negatives for biased classifier")
        
        if len(fp_indices) > 10:
            print(f"Training biased classifier for false positives ({len(fp_indices)} samples)...")
            
            # Sample an equal number of attack instances
            attack_indices = np.where(y_train_binary == 1)[0]
            attack_sampled = np.random.choice(attack_indices, min(len(fp_indices), len(attack_indices)), replace=False)
            
            # Combine FP samples with attack samples
            X_fp = np.vstack((X_train[fp_indices], X_train[attack_sampled]))
            y_fp = np.concatenate((y_train_binary[fp_indices], y_train_binary[attack_sampled]))
            
            # Train the FP biased classifier
            self.biased_classifier_fp = RandomForestClassifier(n_estimators=50, random_state=42)
            self.biased_classifier_fp.fit(X_fp, y_fp)
        else:
            print("Not enough false positives for biased classifier")
    
    def _optimize_kmeans(self, X_train, y_train):
        """
        Optimize the number of clusters for KMeans
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Convert labels to binary (0 for BENIGN, 1 for attacks)
        y_binary = np.array([0 if cls == 'BENIGN' else 1 for cls in y_train])
        
        # Define objective function
        def kmeans_objective(params):
            n_clusters = int(params['n_clusters'])
            
            # Train KMeans with the given number of clusters
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100, random_state=42)
            cluster_labels = kmeans.fit_predict(X_train)
            
            # Determine cluster-to-class mapping
            cluster_to_class = {}
            for cluster_id in range(n_clusters):
                # Get all samples in this cluster
                cluster_samples_idx = np.where(cluster_labels == cluster_id)[0]
                
                if len(cluster_samples_idx) > 0:
                    # Get the binary labels for these samples
                    cluster_binary_labels = y_binary[cluster_samples_idx]
                    
                    # Assign cluster to the majority class
                    if np.mean(cluster_binary_labels) > 0.5:
                        cluster_to_class[cluster_id] = 1  # Attack
                    else:
                        cluster_to_class[cluster_id] = 0  # BENIGN
                else:
                    # If no samples in cluster, assign to BENIGN
                    cluster_to_class[cluster_id] = 0
            
            # Make predictions using the cluster-to-class mapping
            y_pred = np.array([cluster_to_class.get(label, 0) for label in cluster_labels])
            
            # Calculate F1 score
            f1 = f1_score(y_binary, y_pred, average='weighted')
            
            return {'loss': -f1, 'status': STATUS_OK}
        
        # Define search space
        kmeans_space = {
            'n_clusters': hp.quniform('n_clusters', 8, 40, 1)
        }
        
        # Optimize
        best = fmin(fn=kmeans_objective, space=kmeans_space, algo=tpe.suggest, max_evals=10)
        self.n_clusters = int(best.get('n_clusters', N_CLUSTERS))
        
        print(f"Optimal number of clusters: {self.n_clusters}")
    
    def _evaluate_models(self, X_test, y_test):
        """
        Evaluate the trained models on the test set
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        # Get predictions from all models
        y_pred_dt = self.dt_model.predict(X_test)
        y_pred_rf = self.rf_model.predict(X_test)
        y_pred_et = self.et_model.predict(X_test)
        y_pred_xgb = self.xgb_model.predict(X_test)
        
        # Get stacking predictions
        meta_features = np.hstack((
            y_pred_dt.reshape(-1, 1),
            y_pred_rf.reshape(-1, 1),
            y_pred_et.reshape(-1, 1),
            y_pred_xgb.reshape(-1, 1)
        ))
        y_pred_stack = self.stacking_model.predict(meta_features)
        
        # Evaluate each model
        print("\nBase Model Performance:")
        print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
        print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
        print(f"Extra Trees Accuracy: {accuracy_score(y_test, y_pred_et):.4f}")
        print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
        
        print("\nStacking Ensemble Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_stack):.4f}")
        
        # Calculate F1 scores
        f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
        f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
        f1_et = f1_score(y_test, y_pred_et, average='weighted')
        f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
        f1_stack = f1_score(y_test, y_pred_stack, average='weighted')
        
        print("\nF1 Scores (Weighted):")
        print(f"Decision Tree: {f1_dt:.4f}")
        print(f"Random Forest: {f1_rf:.4f}")
        print(f"Extra Trees: {f1_et:.4f}")
        print(f"XGBoost: {f1_xgb:.4f}")
        print(f"Stacking Ensemble: {f1_stack:.4f}")
        
        # Print detailed classification report for the stacking model
        print("\nStacking Ensemble Classification Report:")
        print(classification_report(y_test, y_pred_stack))
        
        # Plot confusion matrix for the stacking model
        cm = confusion_matrix(y_test, y_pred_stack)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=np.unique(y_test),
                   yticklabels=np.unique(y_test))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Stacking Model')
        plt.tight_layout()
        
        # If saving enabled, save the figure instead of displaying
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_baseline(self, df):
        """
        Create a statistical baseline profile from network traffic data
        
        Args:
            df: DataFrame with network traffic features
            
        Returns:
            Dictionary with baseline statistics
        """
        print("\n=== Creating Baseline Profile ===")
        
        # Initialize baseline dictionary
        baseline = {}
        
        # Calculate statistics for each feature
        for column in df.columns:
            # Skip non-numeric or identifier columns
            if column in ['Label', 'Attack Type', 'Binary Label'] or not np.issubdtype(df[column].dtype, np.number):
                continue
            
            try:
                # Calculate statistics
                column_stats = {
                    'mean': float(df[column].mean()),
                    'std': float(df[column].std()),
                    'min': float(df[column].min()),
                    'max': float(df[column].max()),
                    'q1': float(df[column].quantile(0.25)),
                    'median': float(df[column].median()),
                    'q3': float(df[column].quantile(0.75)),
                    # Calculate IQR for outlier detection
                    'iqr': float(df[column].quantile(0.75) - df[column].quantile(0.25))
                }
                
                baseline[column] = column_stats
                
            except Exception as e:
                print(f"Error calculating statistics for column {column}: {e}")
        
        # Add metadata
        baseline['metadata'] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'num_samples': len(df),
            'features': list(df.columns)
        }
        
        self.baseline_profile = baseline
        
        print(f"Created baseline profile with statistics for {len(baseline) - 1} features")
        return baseline
    
    def save(self, output_dir):
        """
        Save the trained model and related information
        
        Args:
            output_dir: Output directory path
        """
        print("\n=== Saving Model ===")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the complete model
        model_path = os.path.join(output_dir, 'mth_ids_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        
        # Save selected features separately
        features_path = os.path.join(output_dir, 'selected_features.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_selector.selected_features, f, indent=4)
        
        # Save feature importance
        if hasattr(self.feature_selector, 'feature_ranks'):
            feature_importance_path = os.path.join(output_dir, 'feature_importance.csv')
            self.feature_selector.feature_ranks.to_csv(feature_importance_path, index=False)
        
        # Save feature importance plot
        plot_path = os.path.join(output_dir, 'feature_importance.png')
        self.feature_selector.plot_feature_importance(plot_path)
        
        # Save the scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save baseline profile if available
        if self.baseline_profile:
            baseline_path = os.path.join(output_dir, 'baseline.json')
            with open(baseline_path, 'w') as f:
                json.dump(self.baseline_profile, f, indent=4)
        
        # Save metadata
        metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'n_clusters': self.n_clusters,
            'n_features': len(self.feature_selector.selected_features),
            'model_versions': {
                'dt': str(self.dt_model),
                'rf': str(self.rf_model),
                'et': str(self.et_model),
                'xgb': str(self.xgb_model)
            }
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Model and related files saved to {output_dir}")
    
    @classmethod
    def load(cls, model_dir):
        """
        Load a trained model from the specified directory
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            Loaded MTH-IDS model
        """
        model_path = os.path.join(model_dir, 'mth_ids_model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded successfully from {model_path}")
        return model


def load_cicids2017(dataset_path):
    """
    Load CICIDS2017 dataset from CSV files
    
    Args:
        dataset_path: Path to dataset directory or file
        
    Returns:
        DataFrame with loaded data
    """
    print("\n=== Loading CICIDS2017 Dataset ===")
    
    # Check if path is a directory or a file
    if os.path.isdir(dataset_path):
        print(f"Loading CSV files from directory: {dataset_path}")
        
        # List all CSV files in the directory
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {dataset_path}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Read and concatenate all CSV files
        df_list = []
        for csv_file in csv_files:
            file_path = os.path.join(dataset_path, csv_file)
            print(f"Reading {file_path}...")
            
            try:
                file_df = pd.read_csv(file_path)
                df_list.append(file_df)
                print(f"  Loaded {len(file_df)} rows, {len(file_df.columns)} columns")
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
                continue
        
        if not df_list:
            raise ValueError("No valid CSV files could be read")
        
        # Concatenate all dataframes
        df = pd.concat(df_list, ignore_index=True)
        
    else:
        # Load single file
        print(f"Loading single CSV file: {dataset_path}")
        df = pd.read_csv(dataset_path)
    
    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


def process_eve_json(eve_json_path):
    """
    Process Suricata EVE JSON file to extract features for baseline
    
    Args:
        eve_json_path: Path to eve.json file
        
    Returns:
        DataFrame with extracted features
    """
    print("\n=== Processing Eve JSON File ===")
    print(f"Reading file: {eve_json_path}")
    
    # Read the file line by line (each line is a separate JSON object)
    data = []
    with open(eve_json_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON
                event = json.loads(line)
                
                # Skip alert records (these are intrusions already detected by Suricata)
                if event.get('event_type') == 'alert':
                    continue
                
                # Add to data list
                data.append(event)
                
                # Print progress for large files
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} lines...")
                    
            except json.JSONDecodeError:
                print(f"Error parsing line {line_num}: Invalid JSON")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    print(f"Processed {len(data)} valid records")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Extract features
    features = extract_features_from_eve(df)
    
    return features


def extract_features_from_eve(df):
    """
    Extract features from Eve JSON records
    
    Args:
        df: DataFrame with Eve JSON records
        
    Returns:
        DataFrame with extracted features
    """
    print("Extracting features from Eve JSON records...")
    
    # Create an empty DataFrame for the features
    features = pd.DataFrame()
    
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
    
    # Fill missing values
    features = features.fillna(0)
    
    print(f"Extracted {len(features.columns)} features")
    
    return features


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NIDS Training Module')
    parser.add_argument('--dataset', required=True, help='Path to CICIDS2017 dataset directory or file')
    parser.add_argument('--model', required=True, help='Path to model output directory')
    parser.add_argument('--baseline', help='Path to eve.json file for baseline creation')
    parser.add_argument('--no-hyperopt', action='store_true', help='Disable hyperparameter optimization')
    
    args = parser.parse_args()
    
    try:
        # Create model output directory
        os.makedirs(args.model, exist_ok=True)
        
        # Load and preprocess dataset
        df = load_cicids2017(args.dataset)
        
        # Create model
        model = MultiTieredHybridIDS()
        
        # Preprocess data
        df = model.preprocess_data(df)
        
        # Map attack categories
        df = model.map_attack_categories(df)
        
        # Train model
        model.train(df, hyperopt=not args.no_hyperopt)
        
        # Create baseline if eve.json is provided
        if args.baseline:
            if os.path.exists(args.baseline):
                # Process eve.json
                features_df = process_eve_json(args.baseline)
                
                # Create baseline profile
                model.create_baseline(features_df)
            else:
                print(f"Warning: Eve JSON file not found at {args.baseline}")
        
        # Save trained model
        model.save(args.model)
        
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
