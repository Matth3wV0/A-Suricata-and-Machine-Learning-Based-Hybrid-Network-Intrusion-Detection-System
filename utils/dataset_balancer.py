import pandas as pd
import logging


# Setup logging
logger = logging.getLogger('hybrid-nids')

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
