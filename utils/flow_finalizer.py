import time
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from dataclasses import asdict
from utils.session_manager import SuricataSession
from utils.adaptive_flow_features import AdaptiveFlowFeatureExtractor
from utils.anomaly_detector import AnomalyDetector

# Setup logging
logger = logging.getLogger('hybrid-nids')

class FlowFinalizer:
    """Processes completed flow sessions and runs anomaly detection"""
    
    def __init__(self, feature_extractor: AdaptiveFlowFeatureExtractor,
                anomaly_detector: AnomalyDetector,
                alert_callback: Optional[Callable] = None,
                min_session_duration: float = 0.0,
                zero_byte_threshold: int = 3,
                save_results: bool = False,
                results_file: str = "flow_results.csv"):
        """Initialize flow finalizer
        
        Args:
            feature_extractor: Feature extractor for flow sessions
            anomaly_detector: Anomaly detector for analyzing flows
            alert_callback: Optional callback function for alerts
            min_session_duration: Minimum session duration to analyze (seconds)
            zero_byte_threshold: Threshold for handling zero-byte flows
            save_results: Whether to save results to CSV
            results_file: Path to results file if saving
        """
        self.feature_extractor = feature_extractor
        self.anomaly_detector = anomaly_detector
        self.alert_callback = alert_callback
        self.min_session_duration = min_session_duration
        self.zero_byte_threshold = zero_byte_threshold
        self.save_results = save_results
        self.results_file = results_file
        
        # Create results file with headers if saving
        if self.save_results:
            with open(self.results_file, 'w') as f:
                f.write("timestamp,flow_id,src_ip,src_port,dst_ip,dst_port,proto,app_proto,")
                f.write("duration,bytes,packets,ml_score,stat_score,combined_score,is_anomalous\n")
        
        # Statistics
        self.stats = {
            'processed_sessions': 0,
            'zero_byte_sessions': 0,
            'short_duration_sessions': 0,
            'analyzed_sessions': 0,
            'anomalous_sessions': 0,
            'alerts_generated': 0
        }
        
        # Keep track of recent zero-byte flows per IP
        self.zero_byte_flows = {}
        self.last_cleanup = time.time()
    
    def process_session(self, session: SuricataSession) -> Dict[str, Any]:
        """Process a finalized session
        
        Args:
            session: Completed SuricataSession object
            
        Returns:
            Dictionary with detection results
        """
        # Convert session to dict if needed
        session_dict = session if isinstance(session, dict) else asdict(session)
        
        # Update statistics
        self.stats['processed_sessions'] += 1
        
        # Check if this is a zero-byte flow
        total_bytes = session_dict.get('total_fwd_bytes', 0) + session_dict.get('total_bwd_bytes', 0)
        if total_bytes == 0:
            self.stats['zero_byte_sessions'] += 1
            result = self._handle_zero_byte_flow(session_dict)
            if result:
                return result
        
        # Check duration
        duration = session_dict.get('duration', 0)
        if duration < self.min_session_duration and self.min_session_duration > 0:
            self.stats['short_duration_sessions'] += 1
            # Continue processing anyway - we'll handle it differently
        
        # Process normally
        self.stats['analyzed_sessions'] += 1
        
        # Extract features
        base_features = self.feature_extractor.extract_from_flow(session)
        
        # Add application-layer features if available
        if hasattr(session, 'get_app_layer_info'):
            app_features = session.get_app_layer_info()
            app_features_df = pd.DataFrame([app_features])
            features = pd.concat([base_features, app_features_df], axis=1)
        else:
            features = base_features
        
        ml_result, stat_result, combined_score = self.anomaly_detector.detect_anomalies(features)
        
        is_anomalous = (ml_result.get('is_anomalous', False) or
                        stat_result.get('is_anomalous', False))
        
        result = {
            'flow_id': session_dict.get('flow_id', ''),
            'timestamp': time.time(),
            'src_ip': session_dict.get('saddr', ''),
            'src_port': session_dict.get('sport', ''),
            'dst_ip': session_dict.get('daddr', ''),
            'dst_port': session_dict.get('dport', ''),
            'proto': session_dict.get('proto', ''),
            'app_proto': session_dict.get('appproto', ''),
            'duration': duration,
            'total_bytes': total_bytes,
            'total_packets': (session_dict.get('total_fwd_packets', 0) + 
                             session_dict.get('total_bwd_packets', 0)),
            'ml_result': ml_result,
            'stat_result': stat_result,
            'combined_score': combined_score,
            'is_anomalous': is_anomalous,
            'session': session_dict  # Include full session for reference
        }
        
        if is_anomalous:
            self.stats['anomalous_sessions'] += 1
            if self.alert_callback:
                self.alert_callback(result)
                self.stats['alerts_generated'] += 1
        
        if self.save_results:
            self._save_result(result)
        
        current_time = time.time()
        if current_time - self.last_cleanup > 300:  # 5 minutes
            self._cleanup_old_data()
            self.last_cleanup = current_time
        
        return result
    
    def _handle_zero_byte_flow(self, session_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Special handling for zero-byte flows to detect brute force patterns
        
        Args:
            session_dict: Session dictionary
            
        Returns:
            Result dictionary if anomalous pattern detected, None otherwise
        """
        # Extract key information
        src_ip = session_dict.get('saddr', '')
        dst_ip = session_dict.get('daddr', '')
        dst_port = session_dict.get('dport', '')
        proto = session_dict.get('proto', '')
        
        if not src_ip or not dst_ip or not dst_port:
            return None
        
        # Create a key for this destination
        dest_key = f"{dst_ip}:{dst_port}:{proto}"
        
        # Initialize tracking for this source IP if needed
        if src_ip not in self.zero_byte_flows:
            self.zero_byte_flows[src_ip] = {}
        
        # Initialize tracking for this destination
        if dest_key not in self.zero_byte_flows[src_ip]:
            self.zero_byte_flows[src_ip][dest_key] = {
                'count': 0,
                'first_seen': time.time(),
                'last_seen': time.time()
            }
        
        # Update tracking
        self.zero_byte_flows[src_ip][dest_key]['count'] += 1
        self.zero_byte_flows[src_ip][dest_key]['last_seen'] = time.time()
        
        # Check for suspicious pattern
        if self.zero_byte_flows[src_ip][dest_key]['count'] >= self.zero_byte_threshold:
            # Calculate rate (flows per second)
            time_span = (self.zero_byte_flows[src_ip][dest_key]['last_seen'] - 
                        self.zero_byte_flows[src_ip][dest_key]['first_seen'])
            
            # Avoid division by zero
            if time_span == 0:
                time_span = 0.1
            
            rate = self.zero_byte_flows[src_ip][dest_key]['count'] / time_span
            
            # Check if rate is suspicious (more than 1 per second)
            if rate > 1.0:
                # Create result for suspicious zero-byte pattern
                result = {
                    'flow_id': session_dict.get('flow_id', ''),
                    'timestamp': time.time(),
                    'src_ip': src_ip,
                    'src_port': session_dict.get('sport', ''),
                    'dst_ip': dst_ip,
                    'dst_port': dst_port,
                    'proto': proto,
                    'app_proto': session_dict.get('appproto', ''),
                    'duration': 0,
                    'total_bytes': 0,
                    'total_packets': 0,
                    'ml_result': {'is_anomalous': True, 'score': 0.8},
                    'stat_result': {'is_anomalous': True, 'score': 0.8},
                    'combined_score': 0.8,
                    'is_anomalous': True,
                    'zero_byte_pattern': True,
                    'zero_byte_count': self.zero_byte_flows[src_ip][dest_key]['count'],
                    'zero_byte_rate': rate,
                    'session': session_dict
                }
                
                # Generate alert
                if self.alert_callback:
                    self.alert_callback(result)
                    self.stats['alerts_generated'] += 1
                
                # Save results if enabled
                if self.save_results:
                    self._save_result(result)
                
                # Reset counter to avoid repeated alerts
                self.zero_byte_flows[src_ip][dest_key]['count'] = 0
                
                return result
        
        return None
    
    def _cleanup_old_data(self) -> None:
        """Clean up old tracking data"""
        current_time = time.time()
        cutoff = current_time - 600  # 10 minutes
        
        # Clean up zero-byte flow tracking
        ips_to_remove = []
        
        for src_ip, destinations in self.zero_byte_flows.items():
            # Clean up old destinations
            dest_to_remove = []
            
            for dest_key, data in destinations.items():
                if data['last_seen'] < cutoff:
                    dest_to_remove.append(dest_key)
            
            # Remove old destinations
            for dest_key in dest_to_remove:
                del destinations[dest_key]
            
            # Mark empty IPs for removal
            if not destinations:
                ips_to_remove.append(src_ip)
        
        # Remove empty IPs
        for src_ip in ips_to_remove:
            del self.zero_byte_flows[src_ip]
    
    def _save_result(self, result: Dict[str, Any]) -> None:
        """Save result to CSV file"""
        try:
            with open(self.results_file, 'a') as f:
                # Create CSV line
                ml_score = result.get('ml_result', {}).get('score', 0)
                stat_score = result.get('stat_result', {}).get('score', 0)
                
                line = (
                    f"{result.get('timestamp', '')},{result.get('flow_id', '')},"
                    f"{result.get('src_ip', '')},{result.get('src_port', '')},"
                    f"{result.get('dst_ip', '')},{result.get('dst_port', '')},"
                    f"{result.get('proto', '')},{result.get('app_proto', '')},"
                    f"{result.get('duration', 0)},{result.get('total_bytes', 0)},"
                    f"{result.get('total_packets', 0)},{ml_score},{stat_score},"
                    f"{result.get('combined_score', 0)},{1 if result.get('is_anomalous', False) else 0}\n"
                )
                
                f.write(line)
        except Exception as e:
            logger.error(f"Error saving result to CSV: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get flow finalizer statistics"""
        return self.stats.copy()
