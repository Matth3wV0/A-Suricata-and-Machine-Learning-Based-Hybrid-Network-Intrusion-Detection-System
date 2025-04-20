#!/usr/bin/env python3
"""
Incremental Flow Analysis Module for Hybrid NIDS
Enables real-time analysis of active flows without waiting for session finalization
"""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

# Setup logging
logger = logging.getLogger('hybrid-nids')

@dataclass
class AnalysisTrigger:
    """Configuration for when to trigger incremental analysis"""
    
    # Time-based trigger (in seconds)
    time_interval: float = 5.0  # Reduced from 10.0 - analyze every 5 seconds
    
    # Packet-based triggers
    packet_threshold: int = 5    # Reduced from 10 - analyze after 5 packets
    packet_increment: int = 10   # Subsequent analysis every 10 more packets
    
    # Byte-based triggers
    byte_threshold: int = 1000   # Reduced from 5000 - analyze after 1KB
    byte_increment: int = 5000   # Subsequent analysis every 5KB more
    
    # Trigger types to use (can combine)
    use_time_trigger: bool = True
    use_packet_trigger: bool = True
    use_byte_trigger: bool = True
    
    # Critical ports that trigger immediate analysis
    critical_ports: Set[int] = field(default_factory=lambda: {
        22,    # SSH
        23,    # Telnet
        3389,  # RDP
        21,    # FTP
        445,   # SMB
        139,   # NetBIOS
        1433,  # MSSQL
        3306,  # MySQL
    })
    
    # Critical protocols that trigger immediate analysis
    critical_protocols: Set[str] = field(default_factory=lambda: {
        'ssh', 'rdp', 'ftp', 'smb', 'telnet'
    })
    
    # Per-port threshold configurations (overrides default thresholds for specific ports)
    # Format: {port: (packet_threshold, byte_threshold, time_interval)}
    port_thresholds: Dict[int, Tuple[int, int, float]] = field(default_factory=lambda: {
        22: (1, 100, 1.0),     # SSH - extremely aggressive thresholds
        23: (1, 100, 1.0),     # Telnet - extremely aggressive thresholds
        21: (2, 200, 1.0),     # FTP - very aggressive thresholds  
        3389: (2, 200, 1.0),   # RDP - very aggressive thresholds
        445: (2, 300, 2.0),    # SMB - aggressive thresholds
        139: (2, 300, 2.0),    # NetBIOS - aggressive thresholds
        1433: (2, 300, 2.0),   # MSSQL - aggressive thresholds
        3306: (2, 300, 2.0),   # MySQL - aggressive thresholds
        53: (3, 300, 2.0),     # DNS - aggressive thresholds
        80: (3, 500, 3.0),     # HTTP - moderately aggressive thresholds
        443: (3, 500, 3.0),    # HTTPS - moderately aggressive thresholds
    })

class IncrementalAnalyzer:
    """
    Analyzes active (non-finalized) sessions in real-time without waiting for flow completion
    """
    
    def __init__(self, feature_extractor, anomaly_detector, alert_callback=None, 
                 config: Optional[AnalysisTrigger] = None):
        """Initialize the incremental analyzer
        
        Args:
            feature_extractor: The feature extractor to use for flow analysis
            anomaly_detector: The anomaly detector to use for predictions
            alert_callback: Function to call when an anomaly is detected
            config: Configuration for analysis triggers
        """
        self.feature_extractor = feature_extractor
        self.anomaly_detector = anomaly_detector
        self.alert_callback = alert_callback
        self.config = config or AnalysisTrigger()
        
        # Track last analysis time for each flow
        self.last_analysis_time: Dict[str, float] = {}
        
        # Track packet counts at last analysis
        self.last_packet_count: Dict[str, int] = {}
        
        # Track byte counts at last analysis
        self.last_byte_count: Dict[str, int] = {}
        
        # Track flows that have already alerted to prevent duplicates
        self.alerted_flows: Set[str] = set()
        
        # Track the last global analysis time
        self.last_global_analysis_time: float = time.time()
        
        # Counter for analyzed flows
        self.analyzed_flow_count: int = 0
        self.anomalous_flow_count: int = 0
        
        # Period for removed closed flows from tracking
        self.cleanup_interval: float = 300  # 5 minutes
        self.last_cleanup_time: float = time.time()
    
    def should_analyze_session(self, session) -> bool:
        """
        Determine if a session should be analyzed - AGGRESSIVE VERSION
        Always returns True for first packet on critical ports or every N packets
        
        Args:
            session: The SuricataSession to check
            
        Returns:
            True if session should be analyzed, False otherwise
        """
        # Get flow ID
        flow_id = session.flow_id
        
        # Don't analyze again if we've already alerted on this flow
        if flow_id in self.alerted_flows:
            return False
        
        # FIRST PACKET ANALYSIS - Always analyze sessions we haven't seen before
        # This ensures immediate analysis for new flows regardless of port/protocol
        if flow_id not in self.last_analysis_time:
            logger.debug(f"First packet analysis for new flow {flow_id}")
            return True
        
        # Check critical criteria - ALWAYS analyze critical services frequently
        is_critical = False
        
        # Check port
        if hasattr(session, 'dport') and session.dport:
            try:
                port = int(session.dport)
                if port in self.config.critical_ports:
                    is_critical = True
            except (ValueError, TypeError):
                pass
        
        # Check protocol
        if hasattr(session, 'appproto') and session.appproto:
            app_proto = session.appproto.lower()
            if app_proto in self.config.critical_protocols:
                is_critical = True
        
        # Get current counts
        current_time = time.time()
        last_time = self.last_analysis_time.get(flow_id, 0)
        total_packets = session.total_fwd_packets + session.total_bwd_packets
        last_packets = self.last_packet_count.get(flow_id, 0)
        
        # CRITICAL SERVICE - Check every packet or at least every second
        if is_critical:
            # For critical services, analyze on EVERY packet change
            if total_packets > last_packets:
                logger.debug(f"Analyzing critical flow {flow_id} - new packets: {total_packets - last_packets}")
                return True
                
            # Also check every second regardless of packet changes
            if current_time - last_time >= 1.0:
                logger.debug(f"Analyzing critical flow {flow_id} - time trigger (1s)")
                return True
        
        # NON-CRITICAL - Still analyze frequently
        # Check every few packets
        if total_packets - last_packets >= 2:  # Every 2 packets
            logger.debug(f"Analyzing flow {flow_id} - packet trigger: {total_packets - last_packets} new packets")
            return True
        
        # Also check every 3 seconds regardless
        if current_time - last_time >= 3.0:
            logger.debug(f"Analyzing flow {flow_id} - time trigger (3s)")
            return True
        
        return False



    def analyze_session(self, session) -> Optional[Dict[str, Any]]:
        """
        Analyze a session incrementally
        
        Args:
            session: The SuricataSession to analyze
            
        Returns:
            Detection result if anomalous, None otherwise
        """
        try:
            flow_id = session.flow_id
            analysis_start_time = time.time()
            logger.info(f"Starting incremental analysis for flow {flow_id} - src={session.saddr}:{session.sport} dst={session.daddr}:{session.dport} proto={session.proto}")

            
            # Calculate detection latency if starttime is available
            detection_latency = None
            start_time_str = None
            
            if hasattr(session, 'starttime') and session.starttime:
                start_time_str = session.starttime
                try:
                    from dateutil import parser
                    import datetime
                    start_time = parser.parse(session.starttime.replace('Z', '+00:00'))
                    current_time = datetime.datetime.now(datetime.timezone.utc)
                    detection_latency = (current_time - start_time).total_seconds()
                    logger.info(f"Flow {flow_id} detection latency: {detection_latency:.2f}s from start time {start_time_str}")
                except Exception as e:
                    logger.warning(f"Failed to calculate detection latency: {e}")
            
            # Determine trigger reason
            trigger_reason = self._determine_trigger_reason(session)
            logger.debug(f"Analyzing flow {flow_id} triggered by: {trigger_reason}")
            
            # Extract features incrementally
            features = self.extract_incremental_features(session)
            
            # Run anomaly detection
            ml_result, stat_result, combined_score = self.anomaly_detector.detect_anomalies(features)
            
            # Update analysis trackers
            self.last_analysis_time[flow_id] = analysis_start_time
            self.last_packet_count[flow_id] = session.total_fwd_packets + session.total_bwd_packets
            self.last_byte_count[flow_id] = session.total_fwd_bytes + session.total_bwd_bytes
            
            # Increment counter
            self.analyzed_flow_count += 1
            
            # Calculate analysis time
            analysis_duration = time.time() - analysis_start_time
            
            # Check for anomaly and generate alert if needed
            is_anomalous = (ml_result.get('is_anomalous', False) or 
                        stat_result.get('is_anomalous', False) or
                        combined_score > 0.7)
            
            if is_anomalous:
                # Mark flow as alerted
                self.alerted_flows.add(flow_id)
                self.anomalous_flow_count += 1
                
                # Get port for logging
                port = None
                if hasattr(session, 'dport'):
                    try:
                        port = int(session.dport)
                    except (ValueError, TypeError):
                        port = session.dport
                
                # Construct result
                result = {
                    'flow_id': flow_id,
                    'timestamp': time.time(),
                    'src_ip': session.saddr,
                    'src_port': session.sport,
                    'dst_ip': session.daddr,
                    'dst_port': session.dport,
                    'proto': session.proto,
                    'app_proto': session.appproto,
                    'duration': time.time() - self.last_analysis_time.get(flow_id, time.time()),
                    'total_bytes': session.total_fwd_bytes + session.total_bwd_bytes,
                    'total_packets': session.total_fwd_packets + session.total_bwd_packets,
                    'ml_result': ml_result,
                    'stat_result': stat_result,
                    'combined_score': combined_score,
                    'is_anomalous': True,
                    'is_incremental': True,  # Flag to indicate this was from incremental analysis
                    'detection_latency': detection_latency,  # Add detection latency information
                    'analysis_duration': analysis_duration,  # Time taken to analyze
                    'trigger_reason': trigger_reason,  # Why this analysis was triggered
                    'start_time': start_time_str,  # Flow start time for reference
                    'session': session  # Include full session for reference
                }
                
                # Log detection details
                logger.info(f"INCREMENTAL ANOMALY DETECTED - Flow {flow_id}")
                logger.info(f"  Source: {session.saddr}:{session.sport} → Destination: {session.daddr}:{session.dport}")
                logger.info(f"  Protocol: {session.proto}/{session.appproto}")
                if port in self.config.critical_ports:
                    logger.info(f"  Critical service: Yes (port {port})")
                logger.info(f"  Detection latency: {detection_latency:.2f}s")
                logger.info(f"  Analysis time: {analysis_duration:.4f}s")
                logger.info(f"  Trigger reason: {trigger_reason}")
                logger.info(f"  Total packets: {session.total_fwd_packets + session.total_bwd_packets}")
                logger.info(f"  Total bytes: {session.total_fwd_bytes + session.total_bwd_bytes}")
                logger.info(f"  ML Score: {ml_result.get('score', 0):.4f}, Statistical Score: {stat_result.get('score', 0):.4f}")
                logger.info(f"  Combined Score: {combined_score:.4f}")
                
                # Generate alert
                if self.alert_callback:
                    self.alert_callback(result)
                
                # And near the end of the method, after analyzing and before returning:
                analysis_duration = time.time() - analysis_start_time
                if is_anomalous:
                    logger.info(f"ANOMALY DETECTED in {analysis_duration:.3f}s - flow {flow_id} score={combined_score:.4f}")
                else:
                    logger.debug(f"No anomaly detected in {analysis_duration:.3f}s - flow {flow_id} score={combined_score:.4f}")

                # Warn about slow analysis
                if analysis_duration > 0.1:  # More than 100ms is concerning
                    logger.warning(f"Slow analysis detected: {analysis_duration:.3f}s for flow {flow_id}")
                

                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error during incremental analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    
    def _determine_trigger_reason(self, session) -> str:
        """Determine the reason why this session was analyzed"""
        flow_id = session.flow_id
        
        # Check if this is the first analysis
        if flow_id not in self.last_analysis_time:
            # Check for critical port or protocol
            if hasattr(session, 'dport'):
                try:
                    port = int(session.dport)
                    if port in self.config.critical_ports:
                        return f"Critical port ({port})"
                except (ValueError, TypeError):
                    pass
            
            if hasattr(session, 'appproto') and session.appproto and session.appproto.lower() in self.config.critical_protocols:
                return f"Critical protocol ({session.appproto})"
                
            return "Initial analysis"
        
        # Check for time-based trigger
        current_time = time.time()
        last_time = self.last_analysis_time.get(flow_id, 0)
        if current_time - last_time >= self.config.time_interval:
            return f"Time interval ({current_time - last_time:.2f}s)"
        
        # Check for packet-based trigger
        total_packets = session.total_fwd_packets + session.total_bwd_packets
        last_packets = self.last_packet_count.get(flow_id, 0)
        if last_packets == 0 and total_packets >= self.config.packet_threshold:
            return f"Packet threshold ({total_packets} packets)"
        elif last_packets > 0 and total_packets - last_packets >= self.config.packet_increment:
            return f"Packet increment ({total_packets - last_packets} new packets)"
        
        # Check for byte-based trigger
        total_bytes = session.total_fwd_bytes + session.total_bwd_bytes
        last_bytes = self.last_byte_count.get(flow_id, 0)
        if last_bytes == 0 and total_bytes >= self.config.byte_threshold:
            return f"Byte threshold ({total_bytes} bytes)"
        elif last_bytes > 0 and total_bytes - last_bytes >= self.config.byte_increment:
            return f"Byte increment ({total_bytes - last_bytes} new bytes)"
        
        return "Unknown trigger"  
    
    def extract_incremental_features(self, session) -> pd.DataFrame:
        """
        Extract features from a partial (active) session
        This is adapted from the standard feature extractor but handles incomplete sessions
        
        Args:
            session: The active SuricataSession to extract features from
            
        Returns:
            DataFrame with extracted features
        """
        try:
            # Use standard feature extractor
            features = self.feature_extractor.extract_from_flow(session)
            
            # Add special flags for incomplete flow
            features['is_incomplete_flow'] = 1  # Flag to indicate partial analysis
            
            # Calculate additional metrics for active flows
            
            # 1. Calculate flow rate metrics over the active duration
            current_time = time.time()
            flow_id = session.flow_id
            first_seen = self.last_analysis_time.get(flow_id, current_time - 60)
            active_duration = max(current_time - first_seen, 0.001)  # Avoid division by zero
            
            # Calculate packet and byte rates since last check
            last_packet_count = self.last_packet_count.get(flow_id, 0)
            last_byte_count = self.last_byte_count.get(flow_id, 0)
            
            current_packet_count = session.total_fwd_packets + session.total_bwd_packets
            current_byte_count = session.total_fwd_bytes + session.total_bwd_bytes
            
            new_packets = current_packet_count - last_packet_count
            new_bytes = current_byte_count - last_byte_count
            
            if active_duration > 0:
                features['recent_packets_per_sec'] = new_packets / active_duration
                features['recent_bytes_per_sec'] = new_bytes / active_duration
            else:
                features['recent_packets_per_sec'] = 0
                features['recent_bytes_per_sec'] = 0
            
            # 2. Add application-layer flags
            if hasattr(session, 'http_event_count') and session.http_event_count > 0:
                features['has_http_activity'] = 1
            else:
                features['has_http_activity'] = 0
                
            if hasattr(session, 'dns_event_count') and session.dns_event_count > 0:
                features['has_dns_activity'] = 1
            else:
                features['has_dns_activity'] = 0
                
            if hasattr(session, 'tls_event_count') and session.tls_event_count > 0:
                features['has_tls_activity'] = 1
            else:
                features['has_tls_activity'] = 0
                
            if hasattr(session, 'ssh_event_count') and session.ssh_event_count > 0:
                features['has_ssh_activity'] = 1
            else:
                features['has_ssh_activity'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting incremental features: {e}")
            
            # Fallback to empty DataFrame with required columns
            empty_df = pd.DataFrame(columns=self.feature_extractor.selected_features)
            empty_df.loc[0] = [0] * len(self.feature_extractor.selected_features)
            return empty_df
    
    def analyze_active_sessions(self, sessions) -> List[Dict[str, Any]]:
        """
        Analyze multiple active sessions
        
        Args:
            sessions: Dictionary of active sessions {flow_id: session}
            
        Returns:
            List of detection results for anomalous sessions
        """
        results = []
        
        # Update last global analysis time
        self.last_global_analysis_time = time.time()
        
        # Cleanup if needed
        if time.time() - self.last_cleanup_time > self.cleanup_interval:
            self.cleanup()
        
        # Analyze each session
        for flow_id, session in sessions.items():
            # Skip flows we've already alerted on
            if flow_id in self.alerted_flows:
                continue
                
            # Check if session should be analyzed
            if self.should_analyze_session(session):
                result = self.analyze_session(session)
                if result:
                    results.append(result)
        
        return results
    
    def cleanup(self) -> None:
        """Clean up tracking data for closed flows"""
        # This is called periodically to remove tracking data for closed or old flows
        current_time = time.time()
        
        # Find old entries to remove (older than cleanup_interval)
        old_flow_ids = [
            flow_id for flow_id, last_time in self.last_analysis_time.items()
            if current_time - last_time > self.cleanup_interval
        ]
        
        # Remove old entries
        for flow_id in old_flow_ids:
            self.last_analysis_time.pop(flow_id, None)
            self.last_packet_count.pop(flow_id, None)
            self.last_byte_count.pop(flow_id, None)
            
            # Only remove from alerted_flows if they're really old
            if current_time - self.last_analysis_time.get(flow_id, 0) > self.cleanup_interval * 3:
                try:
                    self.alerted_flows.remove(flow_id)
                except KeyError:
                    pass
        
        self.last_cleanup_time = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            'active_tracked_flows': len(self.last_analysis_time),
            'alerted_flows': len(self.alerted_flows),
            'analyzed_flow_count': self.analyzed_flow_count,
            'anomalous_flow_count': self.anomalous_flow_count,
            'time_since_last_analysis': time.time() - self.last_global_analysis_time,
            'time_since_last_cleanup': time.time() - self.last_cleanup_time
        }