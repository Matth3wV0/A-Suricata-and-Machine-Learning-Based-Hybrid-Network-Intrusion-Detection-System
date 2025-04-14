#!/usr/bin/env python3
"""
Adaptive Feature Extraction from Suricata Flows and Sessions
This version supports both standard Suricata flows and enriched sessions
"""

import pandas as pd
import numpy as np
import json
import logging
from dataclasses import asdict
from typing import Dict, List, Any, Union, Optional
import datetime
from dateutil import parser

# Setup logging
logger = logging.getLogger('hybrid-nids')

class AdaptiveFlowFeatureExtractor:
    """Extracts features from Suricata flows and sessions with automatic field path detection"""
    
    def __init__(self, selected_features: List[str]):
        self.selected_features = selected_features
        self.field_paths = {}  # Cache for discovered field paths
        self.path_discovery_done = False
        
    def extract_from_flow(self, flow: Any) -> pd.DataFrame:
        """Extract features from a flow object or session with adaptive field path detection"""
        try:
            # Convert flow object to dictionary if it's not already a dict
            flow_dict = flow if isinstance(flow, dict) else asdict(flow)
            
            # Check if this is an enriched session (has app layer info)
            is_enriched_session = 'http_methods' in flow_dict or 'dns_queries' in flow_dict
            
            # Discover field paths if not already done
            if not self.path_discovery_done:
                self._discover_field_paths(flow_dict)
            
            # Extract features based on discovered paths
            features = {}
            
            # 1. Destination Port - direct mapping (dport)
            features['dest_port'] = self._extract_value(flow_dict, 'dest_port', int, 0)
            
            # 2. Flow Duration 
            features['duration'] = self._extract_duration(flow_dict)
            
            # 3. Total Fwd Packets
            features['total_fwd_packets'] = self._extract_value(flow_dict, 'total_fwd_packets', int, 0)
            
            # 4. Total Backward Packets
            features['total_bwd_packets'] = self._extract_value(flow_dict, 'total_bwd_packets', int, 0)
            
            # 5. Total Length of Fwd Packets
            features['total_fwd_bytes'] = self._extract_value(flow_dict, 'total_fwd_bytes', int, 0)
            
            # 6. Total Length of Bwd Packets
            features['total_bwd_bytes'] = self._extract_value(flow_dict, 'total_bwd_bytes', int, 0)
            
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
            
            # Add application layer features if this is an enriched session
            if is_enriched_session:
                app_features = self._extract_app_layer_features(flow_dict)
                features.update(app_features)
            
            # Create DataFrame
            df = pd.DataFrame([features])
            
            # Make sure all required columns are present with correct order
            result_df = pd.DataFrame()
            for feature in self.selected_features:
                if feature in df.columns:
                    result_df[feature] = df[feature]
                else:
                    result_df[feature] = 0
            
            # Add any additional features not in selected_features list
            for col in df.columns:
                if col not in result_df.columns:
                    result_df[col] = df[col]
                    
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
    
    def _extract_app_layer_features(self, flow_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract application layer features from an enriched session"""
        app_features = {}
        
        # HTTP features
        http_methods = flow_dict.get('http_methods', [])
        http_status_codes = flow_dict.get('http_status_codes', [])
        http_hosts = flow_dict.get('http_hosts', [])
        http_event_count = flow_dict.get('http_event_count', 0)
        
        app_features['http_event_count'] = http_event_count
        app_features['has_http'] = 1 if http_event_count > 0 else 0
        
        # Check for HTTP methods
        app_features['has_http_get'] = 1 if 'GET' in http_methods else 0
        app_features['has_http_post'] = 1 if 'POST' in http_methods else 0
        app_features['http_method_count'] = len(http_methods)
        
        # Check for HTTP errors
        has_http_error = False
        has_http_auth_error = False
        for code in http_status_codes:
            if code.startswith('4') or code.startswith('5'):
                has_http_error = True
            if code in ['401', '403', '407']:
                has_http_auth_error = True
        
        app_features['has_http_error'] = 1 if has_http_error else 0
        app_features['has_http_auth_error'] = 1 if has_http_auth_error else 0
        
        # DNS features
        dns_queries = flow_dict.get('dns_queries', [])
        dns_answers = flow_dict.get('dns_answers', [])
        dns_event_count = flow_dict.get('dns_event_count', 0)
        
        app_features['dns_event_count'] = dns_event_count
        app_features['has_dns'] = 1 if dns_event_count > 0 else 0
        app_features['dns_query_count'] = len(dns_queries)
        app_features['dns_answer_count'] = len(dns_answers)
        
        # Check for DNS failures (queries with no answers)
        app_features['has_dns_failure'] = 1 if (dns_event_count > 0 and not dns_answers) else 0
        
        # TLS features
        tls_sni = flow_dict.get('tls_sni', [])
        tls_versions = flow_dict.get('tls_versions', [])
        tls_event_count = flow_dict.get('tls_event_count', 0)
        
        app_features['tls_event_count'] = tls_event_count
        app_features['has_tls'] = 1 if tls_event_count > 0 else 0
        app_features['tls_sni_count'] = len(tls_sni)
        
        # SSH features
        ssh_client_versions = flow_dict.get('ssh_client_versions', [])
        ssh_server_versions = flow_dict.get('ssh_server_versions', [])
        ssh_event_count = flow_dict.get('ssh_event_count', 0)
        
        app_features['ssh_event_count'] = ssh_event_count
        app_features['has_ssh'] = 1 if ssh_event_count > 0 else 0
        
        # Flow state features
        state = flow_dict.get('state', '')
        app_features['is_rejected'] = 1 if state == 'rejected' else 0
        app_features['is_established'] = 1 if state == 'established' else 0
        app_features['is_closed'] = 1 if state in ['closed', 'fin'] else 0
        app_features['is_reset'] = 1 if state == 'rst' else 0
        
        # Combined application features
        app_layer_count = sum([
            1 if http_event_count > 0 else 0,
            1 if dns_event_count > 0 else 0,
            1 if tls_event_count > 0 else 0,
            1 if ssh_event_count > 0 else 0
        ])
        
        app_features['app_layer_count'] = app_layer_count
        app_features['has_mixed_protocols'] = 1 if app_layer_count > 1 else 0
        
        return app_features
    
    def _discover_field_paths(self, flow_dict: Dict[str, Any]) -> None:
        """Discover the paths to relevant fields in the flow dictionary"""
        logger.info("Discovering field paths in Suricata logs...")
        
        # Initialize paths dictionary with potential field locations
        self.field_paths = {
            'dest_port': ['dport', 'dest_port'],
            'duration': ['dur', 'duration', 'flow.duration', 'flow.end-flow.start'],
            'total_fwd_packets': ['spkts', 'pkts_toserver', 'flow.pkts_toserver', 'total_fwd_packets'],
            'total_bwd_packets': ['dpkts', 'pkts_toclient', 'flow.pkts_toclient', 'total_bwd_packets'],
            'total_fwd_bytes': ['sbytes', 'bytes_toserver', 'flow.bytes_toserver', 'total_fwd_bytes'],
            'total_bwd_bytes': ['dbytes', 'bytes_toclient', 'flow.bytes_toclient', 'total_bwd_bytes']
        }
        
        # Check direct fields first
        for feature, paths in self.field_paths.items():
            for path in paths:
                if '.' not in path and path in flow_dict:
                    logger.info(f"Found direct field for {feature}: {path}")
                    self.field_paths[feature] = [path]  # Set this as the primary path
                    break
        
        # Check nested fields under 'flow' if they exist
        if 'flow' in flow_dict and isinstance(flow_dict['flow'], dict):
            for feature, paths in self.field_paths.items():
                for path in paths:
                    # Check if this is a nested path under 'flow'
                    if path.startswith('flow.') and path[5:] in flow_dict['flow']:
                        logger.info(f"Found nested field for {feature}: {path}")
                        self.field_paths[feature] = [path]  # Set this as the primary path
                        break
        
        # Special handling for duration if it's not directly available
        if 'flow' in flow_dict and isinstance(flow_dict['flow'], dict):
            if 'start' in flow_dict['flow'] and 'end' in flow_dict['flow']:
                logger.info("Duration will be calculated from flow.start and flow.end")
                self.field_paths['duration'] = ['flow.start-flow.end']
        
        self.path_discovery_done = True
        logger.info(f"Field path discovery complete: {self.field_paths}")
    
    def _extract_value(self, flow_dict: Dict[str, Any], feature: str, 
                     convert_type=int, default=0) -> Any:
        """Extract a value from the flow dictionary using discovered paths"""
        if feature not in self.field_paths:
            return default
            
        for path in self.field_paths[feature]:
            try:
                # Handle nested paths
                if '.' in path:
                    parts = path.split('.')
                    if parts[0] == 'flow' and len(parts) == 2:
                        # Simple nested path (flow.field)
                        if 'flow' in flow_dict and isinstance(flow_dict['flow'], dict):
                            value = flow_dict['flow'].get(parts[1])
                            if value is not None:
                                return convert_type(value or default)
                    elif '-' in parts[1]:
                        # Special case for duration calculation
                        if path == 'flow.start-flow.end':
                            return self._calculate_duration_from_timestamps(flow_dict)
                # Direct path
                else:
                    value = flow_dict.get(path)
                    if value is not None:
                        return convert_type(value or default)
            except (ValueError, TypeError):
                continue
                
        return default
    
    def _extract_duration(self, flow_dict: Dict[str, Any]) -> float:
        """Extract flow duration with multiple fallback methods"""
        # Try direct duration field first
        duration = self._extract_value(flow_dict, 'duration', float, 0)
        if duration > 0:
            return duration
            
        # Try to calculate from timestamps if duration isn't available
        try:
            if 'starttime' in flow_dict and 'endtime' in flow_dict:
                from datetime import datetime
                # Handle different timestamp formats
                try:
                    start = self._parse_timestamp(flow_dict.get('starttime'))
                    end = self._parse_timestamp(flow_dict.get('endtime'))
                    if start and end:
                        return (end - start).total_seconds()
                except Exception as e:
                    logger.debug(f"Failed to parse timestamps: {e}")
            
            # Try flow object timestamps
            return self._calculate_duration_from_timestamps(flow_dict)
        except:
            return 0
    
    def _calculate_duration_from_timestamps(self, flow_dict: Dict[str, Any]) -> float:
        """Calculate duration from start and end timestamps in the flow object"""
        try:
            if 'flow' in flow_dict and isinstance(flow_dict['flow'], dict):
                flow_obj = flow_dict['flow']
                if 'start' in flow_obj and 'end' in flow_obj:
                    # Timestamps might be unix timestamps or formatted strings
                    start = self._parse_timestamp(flow_obj.get('start'))
                    end = self._parse_timestamp(flow_obj.get('end'))
                    if start and end:
                        return (end - start).total_seconds()
        except Exception as e:
            logger.debug(f"Failed to calculate duration from flow timestamps: {e}")
        return 0
    
    def _parse_timestamp(self, timestamp) -> Optional[datetime.datetime]:
        """Parse a timestamp in various formats"""
        from datetime import datetime
        
        if not timestamp:
            return None
            
        # Try various formats
        try:
            # If it's a unix timestamp (float or int)
            if isinstance(timestamp, (int, float)):
                return datetime.fromtimestamp(timestamp)
                
            # If it's a string
            timestamp_str = str(timestamp)
            
            # Try ISO format (with Z for UTC)
            if 'Z' in timestamp_str:
                return parser.parse(timestamp_str.replace('Z', '+00:00'))
                
            # Try ISO format
            return parser.parse(timestamp_str)
            
        except Exception as e:
            logger.debug(f"Failed to parse timestamp {timestamp}: {e}")
            return None
            
    def _hash_categorical(self, value: Union[str, bool]) -> int:
        """Convert categorical values to numeric using hash"""
        if not value:
            return 0
        return hash(str(value)) % 10000