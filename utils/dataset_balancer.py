import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split

# Setup logging
logger = logging.getLogger('hybrid-nids')

class DatasetBalancer:
    """
    Provides methods to balance datasets for both binary and multi-class classification
    tasks in the NIDS context, addressing the class imbalance issue.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the dataset balancer with random state for reproducibility."""
        self.random_state = random_state
    
    def balance_binary_dataset(self, features: pd.DataFrame, labels: pd.Series, 
                               benign_label: Any = 0, 
                               sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance the dataset for binary classification by undersampling the majority class.
        
        Args:
            features: DataFrame containing the features
            labels: Series containing binary labels
            benign_label: The label value indicating benign/normal traffic
            sample_size: Optional size to sample for each class (default: size of minority class)
            
        Returns:
            Tuple of balanced features and labels
        """
        logger.info("Balancing dataset for binary classification...")
        
        # Separate benign and attack instances
        benign_idx = labels == benign_label
        attack_idx = ~benign_idx
        
        benign_features = features[benign_idx]
        benign_labels = labels[benign_idx]
        attack_features = features[attack_idx]
        attack_labels = labels[attack_idx]
        
        logger.info(f"Original distribution - Benign: {len(benign_labels)}, Attack: {len(attack_labels)}")
        
        # Determine sample size if not specified
        if sample_size is None:
            sample_size = min(len(benign_labels), len(attack_labels))
        
        # Sample both classes
        if len(benign_labels) > sample_size:
            benign_features = benign_features.sample(n=sample_size, random_state=self.random_state)
            benign_labels = benign_labels[benign_features.index]
        
        if len(attack_labels) > sample_size:
            attack_features = attack_features.sample(n=sample_size, random_state=self.random_state)
            attack_labels = attack_labels[attack_features.index]
        
        # Combine and shuffle
        balanced_features = pd.concat([benign_features, attack_features])
        balanced_labels = pd.concat([benign_labels, attack_labels])
        
        # Shuffle the dataset
        shuffle_idx = np.random.RandomState(self.random_state).permutation(len(balanced_labels))
        balanced_features = balanced_features.iloc[shuffle_idx].reset_index(drop=True)
        balanced_labels = balanced_labels.iloc[shuffle_idx].reset_index(drop=True)
        
        logger.info(f"Balanced distribution - Benign: {sum(balanced_labels == benign_label)}, "
                   f"Attack: {sum(balanced_labels != benign_label)}")
        
        return balanced_features, balanced_labels
    
    def balance_multiclass_dataset(self, features: pd.DataFrame, labels: pd.Series, 
                                  min_samples: int = 1000, 
                                  max_samples: int = 5000) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance a multi-class dataset by:
        1. Filtering out classes with very few samples
        2. Undersampling majority classes
        3. Using SMOTE to oversample minority classes
        
        Args:
            features: DataFrame containing the features
            labels: Series containing multi-class labels
            min_samples: Minimum number of samples required for a class to be included
            max_samples: Maximum number of samples to keep for each class
            
        Returns:
            Tuple of balanced features and labels
        """
        logger.info("Balancing dataset for multi-class classification...")
        
        # Check label distribution
        class_counts = labels.value_counts()
        logger.info(f"Original class distribution:\n{class_counts}")
        
        # Filter classes with too few samples
        viable_classes = class_counts[class_counts >= min_samples].index
        logger.info(f"Viable classes: {len(viable_classes)} out of {len(class_counts)}")
        
        # Keep only viable classes
        mask = labels.isin(viable_classes)
        filtered_features = features[mask]
        filtered_labels = labels[mask]
        
        # Sample from each class
        balanced_dfs = []
        
        for class_label in viable_classes:
            class_mask = filtered_labels == class_label
            class_features = filtered_features[class_mask]
            class_labels = filtered_labels[class_mask]
            
            # Undersample if too many
            if len(class_labels) > max_samples:
                class_features = class_features.sample(n=max_samples, random_state=self.random_state)
                class_labels = class_labels[class_features.index]
            
            balanced_dfs.append((class_features, class_labels))
        
        # Combine all classes
        balanced_features = pd.concat([df[0] for df in balanced_dfs])
        balanced_labels = pd.concat([df[1] for df in balanced_dfs])
        
        # Shuffle the dataset
        shuffle_idx = np.random.RandomState(self.random_state).permutation(len(balanced_labels))
        balanced_features = balanced_features.iloc[shuffle_idx].reset_index(drop=True)
        balanced_labels = balanced_labels.iloc[shuffle_idx].reset_index(drop=True)
        
        # Check if imbalanced-learn is available for SMOTE
        try:
            from imblearn.over_sampling import SMOTE
            
            # Apply SMOTE for further balancing
            logger.info("Applying SMOTE to balance minority classes...")
            smote = SMOTE(sampling_strategy='auto', random_state=self.random_state)
            balanced_features_array, balanced_labels_array = smote.fit_resample(balanced_features, balanced_labels)
            
            # Convert back to DataFrame/Series
            balanced_features = pd.DataFrame(balanced_features_array, columns=balanced_features.columns)
            balanced_labels = pd.Series(balanced_labels_array, name=balanced_labels.name)
            
            # Final class distribution
            logger.info(f"Final class distribution after SMOTE:\n{balanced_labels.value_counts()}")
            
        except ImportError:
            logger.warning("imbalanced-learn package not available. SMOTE not applied.")
            logger.warning("To enable SMOTE, install with: pip install imbalanced-learn")
            logger.info(f"Current class distribution without SMOTE:\n{balanced_labels.value_counts()}")
        
        return balanced_features, balanced_labels
    
    def prepare_binary_classification_data(self, df: pd.DataFrame, 
                                          features_col: str = 'features', 
                                          label_col: str = 'Label',
                                          benign_label: Any = 'BENIGN',
                                          test_size: float = 0.25) -> Dict[str, Any]:
        """
        Prepare a balanced dataset specifically for binary classification tasks.
        
        Args:
            df: DataFrame containing features and labels
            features_col: Name of the column containing features (if it's all columns except label_col, set to None)
            label_col: Name of the column containing labels
            benign_label: The label value indicating benign/normal traffic
            test_size: Proportion of the dataset to include in the test split
            
        Returns:
            Dictionary containing train/test splits
        """
        logger.info("Preparing balanced binary classification dataset...")
        
        # Extract features and labels
        if features_col is None:
            X = df.drop(columns=[label_col])
        else:
            X = df[features_col]
        
        y = df[label_col]
        
        # Convert labels to binary (0 for benign, 1 for attack)
        if not np.issubdtype(y.dtype, np.number):
            y = np.where((y == benign_label), 0, 1)
        
        # Balance the dataset
        X_balanced, y_balanced = self.balance_binary_dataset(X, pd.Series(y))
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=test_size, 
            random_state=self.random_state, stratify=y_balanced
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def prepare_multiclass_classification_data(self, df: pd.DataFrame, 
                                             features_col: str = 'features', 
                                             label_col: str = 'Label',
                                             test_size: float = 0.25) -> Dict[str, Any]:
        """
        Prepare a balanced dataset specifically for multi-class classification tasks.
        
        Args:
            df: DataFrame containing features and labels
            features_col: Name of the column containing features (if it's all columns except label_col, set to None)
            label_col: Name of the column containing labels
            test_size: Proportion of the dataset to include in the test split
            
        Returns:
            Dictionary containing train/test splits
        """
        logger.info("Preparing balanced multi-class classification dataset...")
        
        # Extract features and labels
        if features_col is None:
            X = df.drop(columns=[label_col])
        else:
            X = df[features_col]
        
        y = df[label_col]
        
        # Balance the dataset
        X_balanced, y_balanced = self.balance_multiclass_dataset(X, y)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=test_size, 
            random_state=self.random_state, stratify=y_balanced
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }


# Integration functions for the hybrid NIDS system

def integrate_binary_balancing(df, target_col='Label', benign_value=0):
    """
    Integrate binary balancing for the hybrid NIDS system.
    
    Args:
        df: DataFrame containing the dataset
        target_col: The column containing the target variable
        benign_value: The value in target_col that represents benign traffic
        
    Returns:
        Balanced DataFrame
    """
    # Validate inputs
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Count initial class distribution
    class_counts = df[target_col].value_counts()
    logger.info(f"Initial class distribution: {class_counts.to_dict()}")
    
    # Separate benign and attack instances
    benign = df[df[target_col] == benign_value]
    attacks = df[df[target_col] != benign_value]
    
    logger.info(f"Benign samples: {len(benign)}, Attack samples: {len(attacks)}")
    
    # Determine target sample size - default to size of minority class
    # But allow specifying a different balance strategy if needed
    sample_size = min(len(benign), len(attacks))
    logger.info(f"Target sample size per class: {sample_size}")
    
    # Sample both classes to ensure exact balance
    if len(benign) > sample_size:
        benign_sampled = benign.sample(n=sample_size, random_state=42)
    else:
        benign_sampled = benign
        
    if len(attacks) > sample_size:
        attacks_sampled = attacks.sample(n=sample_size, random_state=42)
    else:
        attacks_sampled = attacks
    
    # Combine and shuffle
    balanced_df = pd.concat([benign_sampled, attacks_sampled])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Log final balance check
    final_counts = balanced_df[target_col].value_counts()
    logger.info(f"Final class distribution: {final_counts.to_dict()}")
    
    return balanced_df

def integrate_multiclass_balancing(df, target_col='Attack Type', min_class_size=2000, use_smote=True):
    """
    Integrate multi-class balancing for the hybrid NIDS system.
    
    Args:
        df: DataFrame containing the dataset
        target_col: The column containing the target variable
        min_class_size: Minimum samples per class to include
        use_smote: Whether to use SMOTE for oversampling
        
    Returns:
        Balanced DataFrame
    """
    # Get class counts
    class_counts = df[target_col].value_counts()
    
    # Filter classes with sufficient samples
    viable_classes = class_counts[class_counts >= min_class_size].index
    filtered_df = df[df[target_col].isin(viable_classes)]
    
    # Sample from each class
    dfs = []
    for cls in viable_classes:
        cls_df = filtered_df[filtered_df[target_col] == cls]
        
        # If class is large, downsample
        max_samples = 5000
        if len(cls_df) > max_samples:
            cls_df = cls_df.sample(n=max_samples, random_state=42)
            
        dfs.append(cls_df)
    
    # Combine sampled data
    balanced_df = pd.concat(dfs)
    
    # Apply SMOTE if requested
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            
            X = balanced_df.drop(columns=[target_col])
            y = balanced_df[target_col]
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df[target_col] = y_resampled
            
        except ImportError:
            print("Warning: imbalanced-learn package not available. SMOTE not applied.")
            print("To enable SMOTE, install with: pip install imbalanced-learn")
    
    # Shuffle the final dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df
