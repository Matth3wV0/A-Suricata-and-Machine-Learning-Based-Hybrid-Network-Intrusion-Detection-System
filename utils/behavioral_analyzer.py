#!/usr/bin/env python3
"""
Behavioral Analyzer Module for Hybrid NIDS
Tracks and analyzes network behavior per source IP
"""

import time
import logging
import datetime
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict

# Setup logging
logger = logging.getLogger('hybrid-nids')

@dataclass
class IPBehavior:
    """Class to track behavior metrics for a single IP address"""
    ip_addr: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    # Time window for behavioral metrics (configurable)
    window_size: int = 300  # 5 minutes in seconds
    
    # Tracking flow activity
    total_flows: int = 0
    active_flows: int = 0
    completed_flows: int = 0
    
    # Tracking destination diversity
    dest_ips: Set[str] = field(default_factory=set)
    dest_ports: Set[int] = field(default_factory=set)
    unique_dest_ips_window: List[Tuple[float, str]] = field(default_factory=list)
    unique_dest_ports_window: List[Tuple[float, int]] = field(default_factory=list)
    
    # Tracking protocols
    protocols: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    app_protocols: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Tracking byte and packet volumes
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_packets_sent: int = 0
    total_packets_received: int = 0
    
    # Windows for rate calculations
    bytes_sent_window: List[Tuple[float, int]] = field(default_factory=list)
    packets_sent_window: List[Tuple[float, int]] = field(default_factory=list)
    
    # Application-layer metrics
    http_requests: int = 0
    http_errors: int = 0
    dns_queries: int = 0
    dns_failures: int = 0
    tls_handshakes: int = 0
    tls_failures: int = 0
    ssh_attempts: int = 0
    
    # Connection state tracking
    connection_states: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Scan detection metrics
    port_scan_score: float = 0
    host_scan_score: float = 0
    
    # Brute force metrics
    failed_auth_count: int = 0
    auth_attempt_window: List[Tuple[float, str, int]] = field(default_factory=list)
    
    # Anomaly scores
    overall_anomaly_score: float = 0
    scan_anomaly_score: float = 0
    brute_force_anomaly_score: float = 0
    volume_anomaly_score: float = 0
    
    def update_from_session(self, session_dict: Dict[str, Any]) -> None:
        """Update behavior metrics from a session dictionary"""
        self.last_seen = time.time()
        self.total_flows += 1
        
        # Extract basic session data
        src_ip = session_dict.get('saddr', '')
        dst_ip = session_dict.get('daddr', '')
        try:
            dst_port = int(session_dict.get('dport', 0))
        except (ValueError, TypeError):
            dst_port = 0
            
        proto = session_dict.get('proto', '')
        app_proto = session_dict.get('appproto', '')
        state = session_dict.get('state', '')
        
        # Track connection completion
        is_completed = session_dict.get('is_complete', False)
        if is_completed:
            self.completed_flows += 1
        else:
            self.active_flows += 1
        
        # Update destination diversity
        if dst_ip:
            self.dest_ips.add(dst_ip)
            self.unique_dest_ips_window.append((time.time(), dst_ip))
        
        if dst_port > 0:
            self.dest_ports.add(dst_port)
            self.unique_dest_ports_window.append((time.time(), dst_port))
        
        # Update protocol counters
        if proto:
            self.protocols[proto] += 1
        
        if app_proto:
            self.app_protocols[app_proto] += 1
        
        # Update byte and packet volumes
        fwd_bytes = session_dict.get('total_fwd_bytes', 0)
        bwd_bytes = session_dict.get('total_bwd_bytes', 0)
        fwd_packets = session_dict.get('total_fwd_packets', 0)
        bwd_packets = session_dict.get('total_bwd_packets', 0)
        
        self.total_bytes_sent += fwd_bytes
        self.total_bytes_received += bwd_bytes
        self.total_packets_sent += fwd_packets
        self.total_packets_received += bwd_packets
        
        # Update time windows for rate calculations
        now = time.time()
        self.bytes_sent_window.append((now, fwd_bytes))
        self.packets_sent_window.append((now, fwd_packets))
        
        # Update application layer metrics
        app_info = self._extract_app_layer_info(session_dict)
        
        # HTTP metrics
        self.http_requests += app_info.get('http_requests', 0)
        self.http_errors += app_info.get('http_errors', 0)
        
        # DNS metrics
        self.dns_queries += app_info.get('dns_queries', 0)
        self.dns_failures += app_info.get('dns_failures', 0)
        
        # TLS metrics
        self.tls_handshakes += app_info.get('tls_handshakes', 0)
        self.tls_failures += app_info.get('tls_failures', 0)
        
        # SSH metrics
        self.ssh_attempts += app_info.get('ssh_attempts', 0)
        
        # Track connection states
        if state:
            self.connection_states[state] += 1
        
        # Track authentication failures
        auth_failures = app_info.get('auth_failures', 0)
        if auth_failures > 0:
            self.failed_auth_count += auth_failures
            service = app_proto if app_proto else f"{proto}-{dst_port}"
            self.auth_attempt_window.append((now, service, auth_failures))
        
        # Update scan and behavioral scores
        self._update_behavioral_scores()
    
    def _extract_app_layer_info(self, session_dict: Dict[str, Any]) -> Dict[str, int]:
        """Extract application layer metrics from session data"""
        app_info = {
            'http_requests': 0,
            'http_errors': 0,
            'dns_queries': 0,
            'dns_failures': 0,
            'tls_handshakes': 0,
            'tls_failures': 0,
            'ssh_attempts': 0,
            'auth_failures': 0
        }
        
        # HTTP metrics
        http_event_count = session_dict.get('http_event_count', 0)
        http_status_codes = session_dict.get('http_status_codes', [])
        
        if http_event_count > 0:
            app_info['http_requests'] = http_event_count
            
            # Count HTTP errors (4xx and 5xx status codes)
            for code in http_status_codes:
                if code.startswith('4') or code.startswith('5'):
                    app_info['http_errors'] += 1
                    
                # Count auth failures specifically
                if code == '401' or code == '403' or code == '407':
                    app_info['auth_failures'] += 1
        
        # DNS metrics
        dns_event_count = session_dict.get('dns_event_count', 0)
        dns_answers = session_dict.get('dns_answers', [])
        
        if dns_event_count > 0:
            app_info['dns_queries'] = dns_event_count
            
            # Detect DNS failures (no answers)
            if dns_event_count > 0 and not dns_answers:
                app_info['dns_failures'] += 1
        
        # TLS metrics
        tls_event_count = session_dict.get('tls_event_count', 0)
        
        if tls_event_count > 0:
            app_info['tls_handshakes'] = tls_event_count
            
            # Detect TLS failures based on connection state
            if session_dict.get('state') in ['rejected', 'failed']:
                app_info['tls_failures'] += 1
        
        # SSH metrics
        ssh_event_count = session_dict.get('ssh_event_count', 0)
        
        if ssh_event_count > 0:
            app_info['ssh_attempts'] = ssh_event_count
            
            # Detect SSH failures
            if session_dict.get('state') in ['rejected', 'failed']:
                app_info['auth_failures'] += 1
        
        return app_info
    
    def cleanup_old_data(self) -> None:
        """Remove data outside the time window"""
        now = time.time()
        cutoff = now - self.window_size
        
        # Clean up destination tracking
        self.unique_dest_ips_window = [(t, ip) for t, ip in self.unique_dest_ips_window if t > cutoff]
        self.unique_dest_ports_window = [(t, port) for t, port in self.unique_dest_ports_window if t > cutoff]
        
        # Clean up rate windows
        self.bytes_sent_window = [(t, val) for t, val in self.bytes_sent_window if t > cutoff]
        self.packets_sent_window = [(t, val) for t, val in self.packets_sent_window if t > cutoff]
        
        # Clean up auth attempts
        self.auth_attempt_window = [(t, svc, cnt) for t, svc, cnt in self.auth_attempt_window if t > cutoff]
    
    def _update_behavioral_scores(self) -> None:
        """Update behavioral anomaly scores"""
        self.cleanup_old_data()
        
        # Calculate port scan score
        ports_in_window = len(set(port for _, port in self.unique_dest_ports_window))
        ips_in_window = len(set(ip for _, ip in self.unique_dest_ips_window))
        
        # Port scan detection - many ports, few IPs
        if ips_in_window > 0:
            port_to_ip_ratio = ports_in_window / ips_in_window
            
            # High ratio means many ports per IP
            if port_to_ip_ratio > 10:
                self.port_scan_score = min(1.0, port_to_ip_ratio / 50)
            else:
                self.port_scan_score = 0
        
        # Host scan detection - many IPs, few ports
        if ports_in_window > 0:
            ip_to_port_ratio = ips_in_window / ports_in_window
            
            # High ratio means many IPs per port
            if ip_to_port_ratio > 5:
                self.host_scan_score = min(1.0, ip_to_port_ratio / 20)
            else:
                self.host_scan_score = 0
        
        # Combined scan score
        self.scan_anomaly_score = max(self.port_scan_score, self.host_scan_score)
        
        # Calculate brute force score based on auth failures
        auth_failures_by_service = defaultdict(int)
        for _, service, count in self.auth_attempt_window:
            auth_failures_by_service[service] += count
        
        max_failures = max(auth_failures_by_service.values()) if auth_failures_by_service else 0
        
        # Brute force score based on number of failures to same service
        if max_failures > 3:
            self.brute_force_anomaly_score = min(1.0, max_failures / 20)
        else:
            self.brute_force_anomaly_score = 0
        
        # Calculate volume anomaly score
        bytes_in_window = sum(b for _, b in self.bytes_sent_window)
        packets_in_window = sum(p for _, p in self.packets_sent_window)
        
        # Simple volume-based threshold
        bytes_per_second = bytes_in_window / self.window_size if self.window_size > 0 else 0
        packets_per_second = packets_in_window / self.window_size if self.window_size > 0 else 0
        
        # Volume anomaly based on sustained high rates
        if bytes_per_second > 1000000:  # 1MB/s
            bytes_score = min(1.0, bytes_per_second / 10000000)  # 10MB/s is max score
        else:
            bytes_score = 0
            
        if packets_per_second > 1000:  # 1000 packets/s
            packets_score = min(1.0, packets_per_second / 10000)  # 10000 packets/s is max score
        else:
            packets_score = 0
            
        self.volume_anomaly_score = max(bytes_score, packets_score)
        
        # Overall anomaly score - combine all scores
        self.overall_anomaly_score = max(
            self.scan_anomaly_score,
            self.brute_force_anomaly_score,
            self.volume_anomaly_score
        )
    
    def get_behavioral_features(self) -> Dict[str, float]:
        """Get behavioral features for machine learning"""
        # Ensure scores are up to date
        self._update_behavioral_scores()
        
        window_duration = time.time() - max(self.first_seen, time.time() - self.window_size)
        if window_duration <= 0:
            window_duration = 0.001  # Avoid division by zero
        
        # Calculate features for ML
        features = {
            # Flow metrics
            'flow_rate': self.total_flows / window_duration,
            'complete_flow_ratio': self.completed_flows / max(1, self.total_flows),
            
            # Destination diversity
            'unique_dst_ips': len(self.dest_ips),
            'unique_dst_ports': len(self.dest_ports),
            'dst_ips_per_second': len(set(ip for _, ip in self.unique_dest_ips_window)) / window_duration,
            'dst_ports_per_second': len(set(port for _, port in self.unique_dest_ports_window)) / window_duration,
            
            # Traffic volume
            'bytes_sent_per_second': sum(b for _, b in self.bytes_sent_window) / window_duration,
            'packets_sent_per_second': sum(p for _, p in self.packets_sent_window) / window_duration,
            
            # Application layer
            'http_error_ratio': self.http_errors / max(1, self.http_requests),
            'dns_failure_ratio': self.dns_failures / max(1, self.dns_queries),
            'tls_failure_ratio': self.tls_failures / max(1, self.tls_handshakes),
            'auth_failures_per_second': self.failed_auth_count / window_duration,
            
            # Connection states
            'rejected_ratio': self.connection_states.get('rejected', 0) / max(1, self.total_flows),
            'reset_ratio': self.connection_states.get('rst', 0) / max(1, self.total_flows),
            
            # Anomaly scores
            'port_scan_score': self.port_scan_score,
            'host_scan_score': self.host_scan_score,
            'brute_force_score': self.brute_force_anomaly_score,
            'volume_anomaly_score': self.volume_anomaly_score,
            'overall_anomaly_score': self.overall_anomaly_score
        }
        
        return features


class BehavioralAnalyzer:
    """Analyzes network behavior patterns across multiple flows"""
    
    def __init__(self, window_size: int = 300, cleanup_interval: int = 60,
                max_tracked_ips: int = 10000):
        """Initialize the behavioral analyzer
        
        Args:
            window_size: Size of the time window for behavioral analysis in seconds
            cleanup_interval: Interval between cleanups of expired data in seconds
            max_tracked_ips: Maximum number of IPs to track
        """
        self.window_size = window_size
        self.cleanup_interval = cleanup_interval
        self.max_tracked_ips = max_tracked_ips
        
        # IP behavior tracking
        self.ip_behaviors: Dict[str, IPBehavior] = {}
        
        # Last cleanup time
        self.last_cleanup = time.time()
    
    def process_session(self, session: Any) -> Optional[Dict[str, float]]:
        """Process a finalized session and update behavioral metrics
        
        Args:
            session: A SuricataSession object or dictionary
            
        Returns:
            Behavioral features for the source IP if anomalous, None otherwise
        """
        # Convert session to dict if it's not already
        session_dict = session if isinstance(session, dict) else asdict(session)
        
        # Get source IP
        src_ip = session_dict.get('saddr', '')
        if not src_ip:
            return None
        
        # Create or update IP behavior
        if src_ip not in self.ip_behaviors:
            self.ip_behaviors[src_ip] = IPBehavior(ip_addr=src_ip, window_size=self.window_size)
        
        # Update behavior with session data
        self.ip_behaviors[src_ip].update_from_session(session_dict)
        
        # Check if cleanup is needed
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup()
            self.last_cleanup = current_time
        
        # Check if behavior is anomalous
        anomaly_score = self.ip_behaviors[src_ip].overall_anomaly_score
        if anomaly_score > 0.5:  # Threshold for reporting anomalous behavior
            return self.ip_behaviors[src_ip].get_behavioral_features()
        
        return None
    
    def get_ip_behavior(self, ip_addr: str) -> Optional[IPBehavior]:
        """Get behavior data for a specific IP address"""
        return self.ip_behaviors.get(ip_addr)
    
    def get_top_anomalous_ips(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get the top anomalous IPs by overall anomaly score
        
        Args:
            limit: Maximum number of IPs to return
            
        Returns:
            List of (ip_addr, anomaly_score) tuples
        """
        return sorted(
            [(ip, behavior.overall_anomaly_score) 
             for ip, behavior in self.ip_behaviors.items()],
            key=lambda x: x[1],
            reverse=True
        )[:limit]
    
    def get_scan_activity(self, threshold: float = 0.7) -> List[Tuple[str, float, str]]:
        """Get IPs involved in scanning activity
        
        Args:
            threshold: Minimum scan score to include
            
        Returns:
            List of (ip_addr, scan_score, scan_type) tuples
        """
        scan_ips = []
        
        for ip, behavior in self.ip_behaviors.items():
            if behavior.port_scan_score > threshold:
                scan_ips.append((ip, behavior.port_scan_score, 'port_scan'))
            elif behavior.host_scan_score > threshold:
                scan_ips.append((ip, behavior.host_scan_score, 'host_scan'))
        
        return sorted(scan_ips, key=lambda x: x[1], reverse=True)
    
    def get_brute_force_activity(self, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Get IPs involved in brute force activity
        
        Args:
            threshold: Minimum brute force score to include
            
        Returns:
            List of (ip_addr, brute_force_score) tuples
        """
        return sorted(
            [(ip, behavior.brute_force_anomaly_score) 
             for ip, behavior in self.ip_behaviors.items()
             if behavior.brute_force_anomaly_score > threshold],
            key=lambda x: x[1],
            reverse=True
        )
    
    def cleanup(self) -> None:
        """Clean up expired data and enforce limits"""
        # Clean up each IP's data
        for behavior in self.ip_behaviors.values():
            behavior.cleanup_old_data()
        
        # Remove inactive IPs
        current_time = time.time()
        cutoff = current_time - self.window_size
        
        inactive_ips = [
            ip for ip, behavior in self.ip_behaviors.items()
            if behavior.last_seen < cutoff
        ]
        
        for ip in inactive_ips:
            del self.ip_behaviors[ip]
        
        # Enforce max tracked IPs limit
        if len(self.ip_behaviors) > self.max_tracked_ips:
            # Sort by last_seen (oldest first)
            sorted_ips = sorted(
                self.ip_behaviors.items(),
                key=lambda x: x[1].last_seen
            )
            
            # Keep only the newest IPs
            ips_to_remove = [ip for ip, _ in sorted_ips[:len(self.ip_behaviors) - self.max_tracked_ips]]
            
            for ip in ips_to_remove:
                del self.ip_behaviors[ip]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            'tracked_ips': len(self.ip_behaviors),
            'window_size': self.window_size,
            'cleanup_interval': self.cleanup_interval,
            'time_since_cleanup': time.time() - self.last_cleanup,
            'anomalous_ips': len([ip for ip, behavior in self.ip_behaviors.items() 
                                 if behavior.overall_anomaly_score > 0.5])
        }
