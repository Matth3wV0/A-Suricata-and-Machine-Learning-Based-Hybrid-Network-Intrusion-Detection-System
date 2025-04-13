#!/usr/bin/env python3
"""
Feature extraction from Suricata flows with proper alignment to CICIDS2017 dataset
"""

import pandas as pd
import numpy as np
import json
import logging
from dataclasses import asdict
from typing import Dict, List, Any, Union

# Setup logging
logger = logging.getLogger('hybrid-nids')

class FlowFeatureExtractor:
    """Extracts features from Suricata flows aligned with CICIDS2017 features"""
    
    def __init__(self, selected_features: List[str]):
        self.selected_features = selected_features
        
    def extract_from_flow(self, flow: Any) -> pd.DataFrame:
        """Extract features from a flow object aligned with CICIDS2017 features"""
        try:
            # Convert flow object to dictionary
            flow_dict = asdict(flow)
            
            # Debug log the flow structure
            logger.debug(f"Flow keys: {list(flow_dict.keys())}")
            
            # Extract features based on proper mapping
            features = {}
            
            # 1. Destination Port - direct mapping (dport)
            features['dest_port'] = int(flow_dict.get('dport', 0) or 0)
            
            # 2. Flow Duration - compute from dur field if available, or from starttime/endtime
            if 'dur' in flow_dict and flow_dict.get('dur') is not None and flow_dict.get('dur') != False:
                features['duration'] = float(flow_dict.get('dur', 0) or 0)
            elif 'starttime' in flow_dict and 'endtime' in flow_dict:
                try:
                    from datetime import datetime
                    start = datetime.fromisoformat(flow_dict.get('starttime').replace('Z', '+00:00'))
                    end = datetime.fromisoformat(flow_dict.get('endtime').replace('Z', '+00:00'))
                    features['duration'] = (end - start).total_seconds()
                except Exception as e:
                    logger.warning(f"Failed to compute duration from start/end: {e}")
                    features['duration'] = 0
            else:
                features['duration'] = 0
            
            # 3. Total Fwd Packets - from spkts (pkts_toserver)
            features['total_fwd_packets'] = int(flow_dict.get('spkts', 0) or 0)
            
            # 4. Total Backward Packets - from dpkts (pkts_toclient)
            features['total_bwd_packets'] = int(flow_dict.get('dpkts', 0) or 0)
            
            # 5. Total Length of Fwd Packets - from sbytes (bytes_toserver)
            features['total_fwd_bytes'] = int(flow_dict.get('sbytes', 0) or 0)
            
            # 6. Total Length of Bwd Packets - from dbytes (bytes_toclient)
            features['total_bwd_bytes'] = int(flow_dict.get('dbytes', 0) or 0)
            
            # Derived features (engineered)
            
            # 7. Flow Bytes/s - total bytes divided by duration
            total_bytes = features['total_fwd_bytes'] + features['total_bwd_bytes']
            duration = max(features['duration'], 0.001)  # Avoid division by zero
            features['flow_bytes_per_sec'] = total_bytes / duration
            
            # 8. Flow Packets/s - total packets divided by duration
            total_packets = features['total_fwd_packets'] + features['total_bwd_packets']
            features['flow_packets_per_sec'] = total_packets / duration
            
            # 9. Down/Up Ratio - total bwd bytes divided by fwd bytes
            if features['total_fwd_bytes'] > 0:
                features['down_up_ratio'] = features['total_bwd_bytes'] / features['total_fwd_bytes']
            else:
                features['down_up_ratio'] = 0
            
            # Check if flow record has been correctly parsed - log warning if all zeros
            non_zero_count = sum(1 for val in features.values() if val != 0)
            if non_zero_count <= 1:  # Only dest_port is non-zero
                logger.warning(f"Flow record has all zero features except dest_port: {flow_dict.get('dport')}")
                logger.debug(f"Original flow data: {json.dumps(flow_dict)}")
            
            # Create DataFrame
            df = pd.DataFrame([features])
            
            # Make sure all required columns are present with correct order
            result_df = pd.DataFrame()
            for feature in self.selected_features:
                if feature in df.columns:
                    result_df[feature] = df[feature]
                else:
                    result_df[feature] = 0
                    
            # Log extracted features for debugging
            logger.debug(f"Extracted features: {result_df.to_dict(orient='records')[0]}")
                    
            return result_df
            
        except Exception as e:
            import traceback
            logger.error(f"Error extracting features: {e}")
            logger.error(traceback.format_exc())
            logger.error(f"Flow dict (truncated): {str(flow_dict)[:500]}")
            
            # Return empty DataFrame with correct columns
            empty_df = pd.DataFrame(columns=self.selected_features)
            empty_df.loc[0] = [0] * len(self.selected_features)
            return empty_df
            
    def _hash_categorical(self, value: Union[str, bool]) -> int:
        """Convert categorical values to numeric using hash"""
        if not value:
            return 0
        return hash(str(value)) % 10000

    def debug_flow_dict(self, flow_dict):
        """Debug method to print out flow dictionary structure"""
        logger.debug("Flow Structure Debug:")
        for k, v in flow_dict.items():
            logger.debug(f"  {k}: {type(v).__name__} = {str(v)[:100]}")
            
        # Check if we can find our expected fields
        expected_fields = ['dport', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']
        for field in expected_fields:
            logger.debug(f"Field '{field}' present: {field in flow_dict}, value: {flow_dict.get(field)}")
        
        # Check if flow key exists and what it contains
        if 'flow' in flow_dict:
            logger.debug("Flow sub-object found:")
            flow_obj = flow_dict['flow']
            if isinstance(flow_obj, dict):
                for k, v in flow_obj.items():
                    logger.debug(f"  flow.{k}: {str(v)[:100]}")
        
        return