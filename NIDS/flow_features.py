#!/usr/bin/env python3
"""
Feature extraction from Suricata flows
"""

import pandas as pd
import numpy as np
from dataclasses import asdict
from typing import Dict, List, Any, Union

class FlowFeatureExtractor:
    """Extracts and transforms flow features for the ML models"""
    
    def __init__(self, selected_features: List[str]):
        self.selected_features = selected_features
        
    def extract_from_flow(self, flow: Any) -> pd.DataFrame:
        """Extract features from a flow object"""
        # Convert flow object to dictionary
        flow_dict = asdict(flow)
        
        # Common features across all flow types
        features = {
            'src_port': int(flow_dict.get('sport', 0)),
            'dest_port': int(flow_dict.get('dport', 0)),
            'proto': self._hash_categorical(flow_dict.get('proto', '')),
            'appproto': self._hash_categorical(flow_dict.get('appproto', '')),
        }
        
        # Extract type-specific features
        flow_type = flow_dict.get('type_', '')
        
        if flow_type == 'conn':
            features.update(self._extract_conn_features(flow_dict))
        elif flow_type == 'http':
            features.update(self._extract_http_features(flow_dict))
        elif flow_type == 'ssl':
            features.update(self._extract_ssl_features(flow_dict))
        elif flow_type == 'dns':
            features.update(self._extract_dns_features(flow_dict))
        elif flow_type == 'ssh':
            features.update(self._extract_ssh_features(flow_dict))
        elif flow_type == 'files':
            features.update(self._extract_file_features(flow_dict))
            
        # Ensure all required features are present
        for feature in self.selected_features:
            if feature not in features:
                features[feature] = 0
                
        # Create DataFrame with only selected features in correct order
        df = pd.DataFrame([features])
        available_features = [f for f in self.selected_features if f in df.columns]
        df = df[available_features]
        
        # Fill missing selected features with zeros
        for feature in self.selected_features:
            if feature not in df.columns:
                df[feature] = 0
                
        return df
        
    def _hash_categorical(self, value: Union[str, bool]) -> int:
        """Convert categorical values to numeric using hash"""
        if not value:
            return 0
        return hash(str(value)) % 10000
        
    def _extract_conn_features(self, flow: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from connection flow"""
        features = {}
        
        # Duration-based features
        features['duration'] = float(flow.get('dur', 0))
        
        # Packet-based features
        features['total_fwd_packets'] = int(flow.get('spkts', 0))
        features['total_bwd_packets'] = int(flow.get('dpkts', 0))
        features['total_packets'] = int(flow.get('pkts', 0))
        
        # Byte-based features
        features['total_fwd_bytes'] = int(flow.get('sbytes', 0))
        features['total_bwd_bytes'] = int(flow.get('dbytes', 0))
        features['total_bytes'] = int(flow.get('bytes', 0))
        
        # Derived features
        dur = max(0.001, float(flow.get('dur', 0.001)))  # Avoid division by zero
        features['flow_bytes_per_sec'] = features['total_bytes'] / dur
        features['flow_packets_per_sec'] = features['total_packets'] / dur
        
        # Calculate packet statistics if possible
        if features['total_fwd_packets'] > 0:
            features['avg_fwd_bytes'] = features['total_fwd_bytes'] / features['total_fwd_packets']
        else:
            features['avg_fwd_bytes'] = 0
            
        if features['total_bwd_packets'] > 0:
            features['avg_bwd_bytes'] = features['total_bwd_bytes'] / features['total_bwd_packets']
        else:
            features['avg_bwd_bytes'] = 0
            
        # State features
        features['state'] = self._hash_categorical(flow.get('state', ''))
            
        return features
        
    def _extract_http_features(self, flow: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from HTTP flow"""
        features = {}
        
        # Extract HTTP-specific features
        features['http_method'] = self._hash_categorical(flow.get('method', ''))
        features['http_status_code'] = self._convert_status_code(flow.get('status_code', ''))
        features['http_uri_length'] = len(str(flow.get('uri', '')))
        features['http_host_length'] = len(str(flow.get('host', '')))
        features['http_user_agent_length'] = len(str(flow.get('user_agent', '')))
        features['http_request_body_len'] = int(flow.get('request_body_len', 0))
        features['http_response_body_len'] = int(flow.get('response_body_len', 0))
        
        return features
        
    def _extract_ssl_features(self, flow: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from SSL/TLS flow"""
        features = {}
        
        # Extract SSL/TLS-specific features
        features['ssl_version'] = self._hash_categorical(flow.get('sslversion', ''))
        features['ssl_subject_length'] = len(str(flow.get('subject', '')))
        features['ssl_issuer_length'] = len(str(flow.get('issuer', '')))
        features['ssl_server_name_length'] = len(str(flow.get('server_name', '')))
        
        return features
        
    def _extract_dns_features(self, flow: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from DNS flow"""
        features = {}
        
        # Extract DNS-specific features
        features['dns_query_length'] = len(str(flow.get('query', '')))
        features['dns_answer_count'] = len(flow.get('answers', []))
        features['dns_query_type'] = self._hash_categorical(flow.get('qtype_name', ''))
        
        return features
        
    def _extract_ssh_features(self, flow: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from SSH flow"""
        features = {}
        
        # Extract SSH-specific features
        features['ssh_client_len'] = len(str(flow.get('client', '')))
        features['ssh_server_len'] = len(str(flow.get('server', '')))
        features['ssh_version_len'] = len(str(flow.get('version', '')))
        
        return features
        
    def _extract_file_features(self, flow: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from file flow"""
        features = {}
        
        # Extract file-specific features
        features['file_size'] = int(flow.get('size', 0))
        
        return features
        
    def _convert_status_code(self, status_code: str) -> int:
        """Convert HTTP status code to numeric value"""
        try:
            return int(status_code)
        except (ValueError, TypeError):
            return 0