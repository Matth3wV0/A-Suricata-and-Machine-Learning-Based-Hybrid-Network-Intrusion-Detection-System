#!/usr/bin/env python3
"""
Incremental Flow Analyzer for Hybrid NIDS
Provides continuous analysis of active flows without waiting for session finalization
"""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import asdict
from collections import defaultdict

# Setup logging
logger = logging.getLogger('hybrid-nids')

class IncrementalFlowAnalyzer:
    """
    Performs continuous analysis on active flow sessions for early anomaly detection
    """
    
    def __init__(self, feature_extractor, anomaly_detector, alert_callback=None,
                 analysis_interval=15, min_packets_threshold=10, min_duration_threshold=5.0):
        """
        Initialize the incremental flow analyzer
        
        Args:
            feature_extractor: Feature extractor to use for flow analysis
            anomaly_detector: Anomaly detector for making predictions
            alert_callback: Callback for sending alerts
            analysis_interval: Seconds between analyses of the same flow (default: 15s)
            min_packets_threshold: Minimum packets before analyzing a flow (default: 10)
            min_duration_threshold: Minimum flow duration in seconds before analyzing (default: 5s)
        """
        self.feature_extractor = feature_extractor
        self.anomaly_detector = anomaly_detector
        self.alert_callback = alert_callback
        self.analysis_interval = analysis_interval
        self.min_packets_threshold = min_packets_threshold
        self.min_duration_threshold = min_duration_threshold
        
        # Track analysis timestamps for each flow to prevent too frequent analysis
        self.last_analysis_time = {}
        
        # Store last prediction results to track state changes and prevent alert storms
        self.last_predictions = {}
        
        # Track flows that have already triggered alerts
        self.alerted_flows = set()
        
        # Stats tracking
        self.stats = {
            'total_analyzed': 0,
            'total_alerts': 0,
            'benign_to_malicious': 0,
            'malicious_to_benign': 0,
            'repeated_alerts_prevented': 0
        }
    
    def should_analyze_flow(self, session):
        """
        Determine if a flow should be analyzed based on thresholds and timing
        
        Args:
            session: The flow session object
            
        Returns:
            Boolean indicating whether the flow should be analyzed
        """
        # Skip if recently analyzed (respects the analysis_interval)
        flow_id = session.flow_id
        current_time = time.time()
        
        if flow_id in self.last_analysis_time:
            time_since_last = current_time - self.last_analysis_time[flow_id]
            if time_since_last < self.analysis_interval:
                return False
        
        # Check if the flow meets the minimum thresholds
        total_packets = session.total_fwd_packets + session.total_bwd_packets
        
        # Calculate approximate duration based on timestamps
        if hasattr(session, 'starttime') and session.starttime:
            try:
                from dateutil import parser
                start_time = parser.parse(session.starttime)
                duration = (current_time - start_time.timestamp())
            except:
                # Fallback to using elapsed time since we first saw this flow
                first_seen = self.last_analysis_time.get(flow_id, current_time)
                duration = current_time - first_seen
        else:
            # No timestamp, use elapsed time since first packet
            first_seen = self.last_analysis_time.get(flow_id, current_time)
            duration = current_time - first_seen
        
        # Special handling for protocols that typically have longer sessions
        app_proto = session.appproto if hasattr(session, 'appproto') else ""
        
        # Adjust thresholds based on protocol
        packet_threshold = self.min_packets_threshold
        duration_threshold = self.min_duration_threshold
        
        # For SSH and database connections, we want to analyze earlier
        if app_proto in ['ssh', 'mysql', 'postgres', 'oracle']:
            packet_threshold = max(3, packet_threshold // 2)
            duration_threshold = max(2.0, duration_threshold // 2)
        
        # For HTTP, which tends to have many short flows, we can raise the threshold
        if app_proto == 'http' and not hasattr(session, 'http_event_count'):
            packet_threshold = packet_threshold * 2
        
        # For TLS, wait a bit longer to capture the handshake
        if app_proto == 'tls' and not hasattr(session, 'tls_event_count'):
            packet_threshold = max(5, packet_threshold)
        
        # Check if thresholds are met
        if total_packets >= packet_threshold or duration >= duration_threshold:
            # Update last analysis time
            self.last_analysis_time[flow_id] = current_time
            return True
        
        return False
    
    def analyze_active_flow(self, session):
        """
        Analyze an active flow session for potential anomalies
        
        Args:
            session: The flow session object
            
        Returns:
            Analysis result if anomalous, None otherwise
        """
        # Skip if flow shouldn't be analyzed yet
        if not self.should_analyze_flow(session):
            return None
        
        try:
            print("BBB")
            print(session)
            print("BBB")
            # Convert session to dict if needed
            session_dict = session if isinstance(session, dict) else asdict(session)
            
            # Set a flag to indicate this is a partial analysis
            session_dict['is_complete'] = False
            session_dict['is_partial_analysis'] = True
            
            # Extract flow duration if not already set
            if session_dict.get('duration', 0) == 0 and session_dict.get('starttime'):
                try:
                    from dateutil import parser
                    start_time = parser.parse(session_dict['starttime'])
                    current_time = time.time()
                    session_dict['duration'] = current_time - start_time.timestamp()
                except:
                    # Use time since first packet as fallback
                    first_seen = self.last_analysis_time.get(session_dict['flow_id'], time.time())
                    session_dict['duration'] = time.time() - first_seen
            
            # Check if we have basic flow data
            total_packets = session_dict.get('total_fwd_packets', 0) + session_dict.get('total_bwd_packets', 0)
            if total_packets == 0:
                return None
            
            # Extract features
            features = self.feature_extractor.extract_from_flow(session_dict)
            
            # Run ML models only (skip statistical for partial flows)
            ml_result = self.anomaly_detector.detect_ml_anomaly(features)
            
            # Use a simplified statistic check for partial flows
            # This uses Z-scores but with a higher threshold to reduce false positives
            stat_result = self._simplified_statistics(features)
            
            # Combine scores, but weight ML higher for partial flows
            combined_score = ml_result.get('score', 0) * 0.7 + stat_result.get('score', 0) * 0.3
            
            # Determine if anomalous (ML must detect anomaly for partial flows)
            is_anomalous = ml_result.get('is_anomalous', False)
            
            # Check for state changes - flows becoming anomalous
            flow_id = session_dict['flow_id']
            previous_state = self.last_predictions.get(flow_id, {'is_anomalous': False, 'score': 0})
            
            # Update flow state
            current_state = {
                'is_anomalous': is_anomalous,
                'score': combined_score,
                'timestamp': time.time()
            }
            self.last_predictions[flow_id] = current_state
            
            # Track statistics
            self.stats['total_analyzed'] += 1
            
            # Construct result dictionary
            result = {
                'flow_id': session_dict.get('flow_id', ''),
                'timestamp': time.time(),
                'src_ip': session_dict.get('saddr', ''),
                'src_port': session_dict.get('sport', ''),
                'dst_ip': session_dict.get('daddr', ''),
                'dst_port': session_dict.get('dport', ''),
                'proto': session_dict.get('proto', ''),
                'app_proto': session_dict.get('appproto', ''),
                'duration': session_dict.get('duration', 0),
                'total_bytes': (session_dict.get('total_fwd_bytes', 0) + 
                               session_dict.get('total_bwd_bytes', 0)),
                'total_packets': total_packets,
                'ml_result': ml_result,
                'stat_result': stat_result,
                'combined_score': combined_score,
                'is_anomalous': is_anomalous,
                'is_partial_analysis': True,
                'session': session_dict
            }
            
            # Only generate alert if:
            # 1. Flow is anomalous
            # 2. Either:
            #    a. This is the first time we're detecting an anomaly (benign → malicious)
            #    b. The score has increased significantly since the last alert
            if is_anomalous and self.alert_callback:
                if not previous_state['is_anomalous']:
                    # This is a transition from benign to malicious
                    self.stats['benign_to_malicious'] += 1
                    self.alert_callback(result)
                    self.alerted_flows.add(flow_id)
                    self.stats['total_alerts'] += 1
                elif (flow_id not in self.alerted_flows or 
                      combined_score > previous_state['score'] * 1.5):
                    # Only alert if score increased significantly
                    self.alert_callback(result)
                    self.alerted_flows.add(flow_id)
                    self.stats['total_alerts'] += 1
                else:
                    # Prevent alert storm
                    self.stats['repeated_alerts_prevented'] += 1
            
            # If a previously malicious flow becomes benign, track the transition
            if previous_state['is_anomalous'] and not is_anomalous:
                self.stats['malicious_to_benign'] += 1
            
            # Return result if anomalous
            if is_anomalous:
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error in incremental flow analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _simplified_statistics(self, features):
        """
        Perform simplified statistical analysis on partial flow data
        
        Args:
            features: DataFrame containing flow features
            
        Returns:
            Dictionary with simplified statistical analysis results
        """
        try:
            # Initialize with safe defaults
            score = 0.0
            is_anomalous = False
            details = []
            
            # Check for baseline statistics in anomaly detector
            if not hasattr(self.anomaly_detector, 'baseline') or not self.anomaly_detector.baseline:
                return {'score': score, 'is_anomalous': is_anomalous, 'details': details}
            
            # Identify key features for quick analysis
            key_features = [
                'flow_bytes_per_sec', 
                'flow_packets_per_sec',
                'down_up_ratio'
            ]
            
            # Only use features available in both the flow and baseline
            available_features = [f for f in key_features 
                               if f in features.columns and f in self.anomaly_detector.baseline]
            
            if not available_features:
                return {'score': score, 'is_anomalous': is_anomalous, 'details': details}
            
            # Check each feature with a high threshold to reduce false positives
            z_threshold = 5.0  # Using higher threshold for partial flows
            
            for feature in available_features:
                if len(features) == 0:
                    continue
                    
                # Get feature value
                value = float(features[feature].iloc[0])
                
                # Get baseline stats
                stats = self.anomaly_detector.baseline[feature]
                
                # Skip features with no variation in baseline
                if stats['std'] <= 0:
                    continue
                
                # Calculate Z-score
                z_score = (value - stats['mean']) / stats['std']
                
                # Add to score if exceeds threshold
                if abs(z_score) > z_threshold:
                    # Use a reduced weight for statistics on partial flows
                    feature_weight = 0.5
                    score += abs(z_score) * feature_weight / len(available_features)
                    
                    # Add to details
                    details.append({
                        'feature': feature,
                        'value': float(value),
                        'z_score': float(z_score),
                        'baseline_mean': stats['mean'],
                        'baseline_std': stats['std']
                    })
            
            # Determine if anomalous - using a conservative approach for partial flows
            # Only flag as anomalous if score is very high or multiple features are anomalous
            is_anomalous = score > 3.0 or len(details) >= 2
            
            return {
                'score': min(1.0, score / 5.0),  # Normalize score to [0,1]
                'is_anomalous': is_anomalous,
                'details': details
            }
            
        except Exception as e:
            logger.error(f"Error in simplified statistics: {e}")
            return {'score': 0.0, 'is_anomalous': False, 'details': []}
    
    def analyze_active_flows(self, active_sessions):
        """
        Analyze a batch of active flow sessions
        
        Args:
            active_sessions: List of active session objects
            
        Returns:
            List of anomalous flow results
        """
        results = []
        
        for session in active_sessions:
            result = self.analyze_active_flow(session)
            if result:
                results.append(result)
        
        return results
    
    def cleanup(self):
        """Clean up old tracking data"""
        # Remove old analysis timestamps (older than 30 minutes)
        current_time = time.time()
        cutoff = current_time - 1800  # 30 minutes
        
        old_flow_ids = [flow_id for flow_id, last_time in self.last_analysis_time.items()
                       if last_time < cutoff]
        
        for flow_id in old_flow_ids:
            self.last_analysis_time.pop(flow_id, None)
            self.last_predictions.pop(flow_id, None)
            
        # Clear alerted flows that are old
        self.alerted_flows = {flow_id for flow_id in self.alerted_flows 
                            if flow_id in self.last_analysis_time}
    
    def get_stats(self):
        """Get analyzer statistics"""
        return self.stats.copy()