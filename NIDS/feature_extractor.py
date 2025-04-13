#!/usr/bin/env python3
"""
Improved Feature Extraction for Suricata Flows
"""

import pandas as pd
import numpy as np
from dataclasses import asdict
from typing import Dict, List, Any, Union

class FlowFeatureExtractor:
    """Extracts and transforms flow features aligned with CICIDS2017 dataset"""
    
    def __init__(self, selected_features: List[str]):
        self.selected_features = selected_features
        
    def extract_from_flow(self, flow: Any) -> pd.DataFrame:
        """Extract features from a flow object, aligned with CICIDS2017 features"""
        try:
            # Convert flow object to dictionary
            flow_dict = asdict(flow)
            print(flow_dict)
            # Base features (all flow types)
            features = {
                # Destination Port (comes from dport in Suricata)
                'dest_port': int(flow_dict.get('dport', 0) or 0),
                
                # Flow Duration (comes from dur in Suricata)
                'duration': float(flow_dict.get('dur', 0) or 0.001),
                
                # Total Fwd Packets (comes from spkts in Suricata)
                'total_fwd_packets': int(flow_dict.get('pkts_toserver', 0) or 0),
                
                # Total Backward Packets (comes from dpkts in Suricata)
                'total_bwd_packets': int(flow_dict.get('pkts_toclient', 0) or 0),
                
                # Total Length of Fwd Packets (comes from sbytes in Suricata)
                'total_fwd_bytes': int(flow_dict.get('bytes_toserver', 0) or 0),
                
                # Total Length of Bwd Packets (comes from dbytes in Suricata)
                'total_bwd_bytes': int(flow_dict.get('bytes_toclient', 0) or 0)
            }
            
            # Derived features
            
            # Flow Bytes/s (total_bytes / duration)
            # Avoid division by zero by using max(duration, 0.001)
            dur = max(float(flow_dict.get('dur', 0) or 0.001), 0.001)
            total_bytes = int(flow_dict.get('sbytes', 0) or 0) + int(flow_dict.get('dbytes', 0) or 0)
            features['flow_bytes_per_sec'] = total_bytes / dur
            
            # Flow Packets/s (total_packets / duration)
            total_packets = int(flow_dict.get('spkts', 0) or 0) + int(flow_dict.get('dpkts', 0) or 0)
            features['flow_packets_per_sec'] = total_packets / dur
            
            # Down/Up Ratio (dbytes / sbytes)
            sbytes = int(flow_dict.get('sbytes', 0) or 0)
            dbytes = int(flow_dict.get('dbytes', 0) or 0)
            # Avoid division by zero
            if sbytes > 0:
                features['down_up_ratio'] = dbytes / sbytes
            else:
                features['down_up_ratio'] = 0
            
            # Create DataFrame with only selected features
            df = pd.DataFrame([features])
            
            # Filter to only include selected features
            df = df[[col for col in self.selected_features if col in df.columns]]
            
            # Fill missing selected features with zeros
            for feature in self.selected_features:
                if feature not in df.columns:
                    df[feature] = 0
                    
            return df
        except Exception as e:
            import traceback
            print(f"Error extracting features: {str(e)}")
            print(traceback.format_exc())
            # Return empty DataFrame with correct columns
            empty_df = pd.DataFrame(columns=self.selected_features)
            # Add a row of zeros
            empty_df.loc[0] = [0] * len(self.selected_features)
            return empty_df
        
    def _hash_categorical(self, value: Union[str, bool]) -> int:
        """Convert categorical values to numeric using hash"""
        if not value:
            return 0
        return hash(str(value)) % 10000
        
    def _extract_additional_features(self, flow_dict: Dict[str, Any]) -> Dict[str, float]:
        """Extract additional features if needed for specific flow types"""
        features = {}
        flow_type = flow_dict.get('type_', '')
        
        if flow_type == 'http':
            features['http_method'] = self._hash_categorical(flow_dict.get('method', ''))
            features['http_status_code'] = self._convert_status_code(flow_dict.get('status_code', ''))
            features['http_uri_length'] = len(str(flow_dict.get('uri', '')))
        
        elif flow_type == 'dns':
            features['dns_query_length'] = len(str(flow_dict.get('query', '')))
            features['dns_answer_count'] = len(flow_dict.get('answers', []))
            
        elif flow_type == 'ssl':
            features['ssl_subject_length'] = len(str(flow_dict.get('subject', '')))
            features['ssl_server_name_length'] = len(str(flow_dict.get('server_name', '')))
        
        return features
        
    def _convert_status_code(self, status_code: str) -> int:
        """Convert HTTP status code to numeric value"""
        try:
            return int(status_code)
        except (ValueError, TypeError):
            return 0