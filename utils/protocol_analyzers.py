#!/usr/bin/env python3
"""
Protocol-specific analyzers for immediate detection in various network protocols
"""

import time
import re
import logging
from typing import Dict, List, Set, Any, Callable, Optional
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger('hybrid-nids')

@dataclass
class DetectionResult:
    """Result from protocol analysis for alert generation"""
    is_detected: bool = False
    alert_type: str = ""
    count: int = 0
    time_span: float = 0.0
    score: float = 0.0
    details: str = ""

class ProtocolAnalyzer:
    """Analyzes events from different protocols for immediate detection"""
    
    def __init__(self, alert_callback: Optional[Callable] = None):
        """Initialize protocol analyzer
        
        Args:
            alert_callback: Function to call when detection occurs
        """
        self.alert_callback = alert_callback
        
        # Initialize tracking structures
        self._http_tracking = {
            'ip_requests': {},
            'suspicious_uris': [
                '/wp-login.php', '/admin/', '/phpMyAdmin/', '/administrator/',  # Common admin paths
                '.php?id=', '.asp?id=', 'select+from', 'union+select',          # SQL injection patterns
                'eval(', 'exec(', 'system(', 'passthru(',                       # Code injection patterns
                '../', '..%2f', '.git/', '.env',                                # Path traversal & sensitive files
                'wp-config', 'config.php', '.bak', '.old'                       # Config/backup files
            ],
            'suspicious_agents': [
                'sqlmap', 'nikto', 'nmap', 'masscan', 'zgrab',                  # Scanner tools
                'curl/', 'wget/', 'python-requests/', 'go-http-client/'         # Automation tools
            ]
        }
        
        self._dns_tracking = {
            'ip_queries': {},
            'domain_requests': {},
            'suspicious_patterns': [
                r'[a-z0-9]{30,}',  # Very long domain parts (possible DGA)
                r'\.top$|\.xyz$|\.pw$',  # TLDs commonly used for malware
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP addresses in DNS queries
            ]
        }
        
        self._ssh_tracking = {
            'connection_attempts': {},
            'suspicious_clients': [
                'libssh', 'paramiko', 'putty', 'dropbear',  # Common tools used in automation
                'nmap', 'metasploit', 'hydra'  # Known scanning/attack tools
            ]
        }
        
        self._tls_tracking = {
            'connection_attempts': {},
            'domain_requests': {},
            'suspicious_patterns': [
                r'[a-z0-9]{30,}\.',  # Very long domain parts
                r'\.top$|\.xyz$|\.pw$',  # TLDs commonly used for malware
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP addresses in SNI
            ],
            'weak_versions': ['SSLv2', 'SSLv3', 'TLSv1.0']
        }
        
        # Track last alert times to implement rate limiting
        self._last_alert_time = {}
        
        # Track last cleanup time
        self._last_cleanup = time.time()
    
    def analyze_http_event(self, event) -> DetectionResult:
        """Immediate analysis of HTTP events for suspicious patterns."""
        if not hasattr(event, 'saddr') or not hasattr(event, 'daddr'):
            return DetectionResult()
        
        src_ip = event.saddr
        dst_ip = event.daddr
        dst_port = getattr(event, 'dport', '80')
        uri = getattr(event, 'uri', '')
        method = getattr(event, 'method', '')
        status = getattr(event, 'status_code', '')
        host = getattr(event, 'host', '')
        user_agent = getattr(event, 'user_agent', '')
        
        # Track requests per IP
        if src_ip not in self._http_tracking['ip_requests']:
            self._http_tracking['ip_requests'][src_ip] = {
                'count': 0,
                'first_seen': time.time(),
                'last_seen': time.time(),
                'methods': {},
                'status_codes': {},
                'paths': set(),
                'user_agents': set()
            }
        
        ip_stats = self._http_tracking['ip_requests'][src_ip]
        ip_stats['count'] += 1
        ip_stats['last_seen'] = time.time()
        
        if method:
            ip_stats['methods'][method] = ip_stats['methods'].get(method, 0) + 1
        
        if status:
            ip_stats['status_codes'][status] = ip_stats['status_codes'].get(status, 0) + 1
        
        if uri:
            ip_stats['paths'].add(uri)
        
        if user_agent:
            ip_stats['user_agents'].add(user_agent)
        
        # DETECTION LOGIC
        
        # 1. High request rate (possible DoS)
        time_window = ip_stats['last_seen'] - ip_stats['first_seen']
        requests_per_second = ip_stats['count'] / max(time_window, 1)
        
        if requests_per_second > 10 and ip_stats['count'] > 30:
            # More than 10 requests per second - possible DoS
            return DetectionResult(
                is_detected=True,
                alert_type='http_attack',
                count=ip_stats['count'],
                time_span=time_window,
                score=0.75,
                details=f"High HTTP request rate: {requests_per_second:.1f} req/sec"
            )
        
        # 2. High error rate (client or server errors)
        error_count = sum(ip_stats['status_codes'].get(str(code), 0) for code in range(400, 600))
        if error_count > 10 and time_window <= 60:
            # More than 10 errors in a minute - possible scanning/brute force
            return DetectionResult(
                is_detected=True,
                alert_type='http_attack',
                count=error_count,
                time_span=time_window,
                score=0.80,
                details=f"High HTTP error rate: {error_count} errors"
            )
        
        # 3. Too many unique paths (directory scanning)
        if len(ip_stats['paths']) > 20 and time_window <= 60:
            # More than 20 unique paths in a minute - possible directory scanning
            return DetectionResult(
                is_detected=True,
                alert_type='http_attack',
                count=len(ip_stats['paths']),
                time_span=time_window,
                score=0.85,
                details=f"Directory scanning detected: {len(ip_stats['paths'])} unique paths"
            )
        
        # 4. Check for suspicious URIs
        if uri:
            for pattern in self._http_tracking['suspicious_uris']:
                if pattern in uri.lower():
                    # Suspicious URI pattern detected
                    return DetectionResult(
                        is_detected=True,
                        alert_type='http_attack',
                        count=1,
                        time_span=0.1,
                        score=0.90,
                        details=f"Suspicious URI detected: {uri} (matched pattern: {pattern})"
                    )
        
        # 5. Check for suspicious user agents
        if user_agent:
            for agent in self._http_tracking['suspicious_agents']:
                if agent.lower() in user_agent.lower():
                    # Suspicious user agent detected
                    return DetectionResult(
                        is_detected=True,
                        alert_type='http_attack',
                        count=1,
                        time_span=0.1,
                        score=0.85,
                        details=f"Suspicious user agent: {user_agent}"
                    )
        
        return DetectionResult()
    
    def analyze_dns_event(self, event) -> DetectionResult:
        """Immediate analysis of DNS events for suspicious patterns."""
        if not hasattr(event, 'query') or not hasattr(event, 'saddr') or not hasattr(event, 'daddr'):
            return DetectionResult()
        
        src_ip = event.saddr
        dst_ip = event.daddr
        query = event.query
        
        # Track queries per source IP
        if src_ip not in self._dns_tracking['ip_queries']:
            self._dns_tracking['ip_queries'][src_ip] = {
                'count': 0,
                'first_seen': time.time(),
                'last_seen': time.time(),
                'unique_domains': set(),
                'nxdomains': 0
            }
        
        ip_stats = self._dns_tracking['ip_queries'][src_ip]
        ip_stats['count'] += 1
        ip_stats['last_seen'] = time.time()
        
        if query:
            ip_stats['unique_domains'].add(query)
            
            # Check for NXDomain (failed resolution)
            if not event.answers or len(event.answers) == 0:
                ip_stats['nxdomains'] += 1
        
        # Track domains being queried
        if query and query not in self._dns_tracking['domain_requests']:
            self._dns_tracking['domain_requests'][query] = {
                'count': 0,
                'first_seen': time.time(),
                'sources': set()
            }
        
        if query:
            domain_stats = self._dns_tracking['domain_requests'][query]
            domain_stats['count'] += 1
            domain_stats['sources'].add(src_ip)
        
        # DETECTION LOGIC
        
        # 1. High query rate from a single source
        time_window = ip_stats['last_seen'] - ip_stats['first_seen']
        if ip_stats['count'] > 50 and time_window <= 60:
            # More than 50 queries in a minute - possible DNS tunneling or flood
            return DetectionResult(
                is_detected=True,
                alert_type='dns_attack',
                count=ip_stats['count'],
                time_span=time_window,
                score=0.80,
                details=f"High DNS query rate: {ip_stats['count']} queries in {time_window:.1f}s"
            )
        
        # 2. High number of unique domains from a single source
        if len(ip_stats['unique_domains']) > 30 and time_window <= 60:
            # More than 30 unique domains in a minute - possible C2 or data exfiltration
            return DetectionResult(
                is_detected=True,
                alert_type='dns_attack',
                count=len(ip_stats['unique_domains']),
                time_span=time_window,
                score=0.85,
                details=f"Possible data exfiltration: {len(ip_stats['unique_domains'])} unique domains"
            )
        
        # 3. High NXDOMAIN rate (failed resolutions) - possible DGA
        if ip_stats['nxdomains'] > 15 and time_window <= 60:
            # More than 15 failed resolutions in a minute - possible domain generation algorithm
            return DetectionResult(
                is_detected=True,
                alert_type='dns_attack',
                count=ip_stats['nxdomains'],
                time_span=time_window,
                score=0.90,
                details=f"Possible DGA: {ip_stats['nxdomains']} failed DNS resolutions"
            )
        
        # 4. Check for suspicious domain patterns
        if query:
            for pattern in self._dns_tracking['suspicious_patterns']:
                if re.search(pattern, query):
                    # Suspicious domain pattern detected
                    return DetectionResult(
                        is_detected=True,
                        alert_type='dns_attack',
                        count=1,
                        time_span=0.1,
                        score=0.85,
                        details=f"Suspicious DNS query: {query}"
                    )
        
        return DetectionResult()
    
    def analyze_ssh_event(self, event) -> DetectionResult:
        """Immediate analysis of SSH events for suspicious patterns."""
        if not hasattr(event, 'saddr') or not hasattr(event, 'daddr'):
            return DetectionResult()
        
        src_ip = event.saddr
        dst_ip = event.daddr
        dst_port = getattr(event, 'dport', '22')
        client_version = getattr(event, 'client', '')
        server_version = getattr(event, 'server', '')
        
        # Create key for this source-destination pair
        key = f"{src_ip}:{dst_ip}:{dst_port}"
        
        # Track connection attempts
        if key not in self._ssh_tracking['connection_attempts']:
            self._ssh_tracking['connection_attempts'][key] = {
                'count': 0,
                'first_seen': time.time(),
                'last_seen': time.time(),
                'client_versions': set()
            }
        
        conn_stats = self._ssh_tracking['connection_attempts'][key]
        conn_stats['count'] += 1
        conn_stats['last_seen'] = time.time()
        
        if client_version:
            conn_stats['client_versions'].add(client_version)
        
        # DETECTION LOGIC
        
        # 1. High connection rate (brute force)
        time_window = conn_stats['last_seen'] - conn_stats['first_seen']
        if conn_stats['count'] >= 5 and time_window <= 60:
            # 5 or more connections in a minute - possible brute force
            return DetectionResult(
                is_detected=True,
                alert_type='ssh_brute_force',
                count=conn_stats['count'],
                time_span=time_window,
                score=0.90,
                details=f"Possible SSH brute force: {conn_stats['count']} attempts in {time_window:.1f}s"
            )
        
        # 2. Multiple client versions from same source (suspicious behavior)
        if len(conn_stats['client_versions']) >= 3:
            # Multiple client versions - possible scanning/tool switching
            return DetectionResult(
                is_detected=True,
                alert_type='ssh_brute_force',
                count=len(conn_stats['client_versions']),
                time_span=time_window,
                score=0.85,
                details=f"Multiple SSH client versions: {', '.join(conn_stats['client_versions'])}"
            )
        
        # 3. Check for suspicious client software
        if client_version:
            client_lower = client_version.lower()
            for suspicious in self._ssh_tracking['suspicious_clients']:
                if suspicious.lower() in client_lower:
                    # Suspicious client detected
                    return DetectionResult(
                        is_detected=True,
                        alert_type='ssh_brute_force',
                        count=1,
                        time_span=0.1,
                        score=0.95,
                        details=f"Suspicious SSH client: {client_version}"
                    )
        
        return DetectionResult()
    
    def analyze_tls_event(self, event) -> DetectionResult:
        """Immediate analysis of TLS events for suspicious patterns."""
        if not hasattr(event, 'saddr') or not hasattr(event, 'daddr'):
            return DetectionResult()
        
        src_ip = event.saddr
        dst_ip = event.daddr
        dst_port = getattr(event, 'dport', '443')
        sni = getattr(event, 'server_name', '')
        version = getattr(event, 'sslversion', '')
        subject = getattr(event, 'subject', '')
        
        # Track connections by source IP
        if src_ip not in self._tls_tracking['connection_attempts']:
            self._tls_tracking['connection_attempts'][src_ip] = {
                'count': 0,
                'first_seen': time.time(),
                'last_seen': time.time(),
                'unique_destinations': set(),
                'unique_domains': set(),
                'versions': set()
            }
        
        conn_stats = self._tls_tracking['connection_attempts'][src_ip]
        conn_stats['count'] += 1
        conn_stats['last_seen'] = time.time()
        conn_stats['unique_destinations'].add(f"{dst_ip}:{dst_port}")
        
        if sni:
            conn_stats['unique_domains'].add(sni)
        
        if version:
            conn_stats['versions'].add(version)
        
        # Track SNI domains
        if sni and sni not in self._tls_tracking['domain_requests']:
            self._tls_tracking['domain_requests'][sni] = {
                'count': 0,
                'first_seen': time.time(),
                'sources': set()
            }
        
        if sni:
            domain_stats = self._tls_tracking['domain_requests'][sni]
            domain_stats['count'] += 1
            domain_stats['sources'].add(src_ip)
        
        # DETECTION LOGIC
        
        # 1. High connection rate to different destinations (scanning)
        time_window = conn_stats['last_seen'] - conn_stats['first_seen']
        if len(conn_stats['unique_destinations']) > 20 and time_window <= 60:
            # More than 20 unique destinations in a minute - possible scanning
            return DetectionResult(
                is_detected=True,
                alert_type='tls_attack',
                count=len(conn_stats['unique_destinations']),
                time_span=time_window,
                score=0.85,
                details=f"TLS scanning detected: {len(conn_stats['unique_destinations'])} destinations"
            )
        
        # 2. Check for use of weak/outdated TLS versions
        if version in self._tls_tracking['weak_versions']:
            # Old/weak TLS version detected
            return DetectionResult(
                is_detected=True,
                alert_type='tls_attack',
                count=1,
                time_span=0.1,
                score=0.75,
                details=f"Weak TLS version: {version}"
            )
        
        # 3. Check for suspicious SNI patterns
        if sni:
            for pattern in self._tls_tracking['suspicious_patterns']:
                if re.search(pattern, sni):
                    # Suspicious SNI pattern detected
                    return DetectionResult(
                        is_detected=True,
                        alert_type='tls_attack',
                        count=1,
                        time_span=0.1,
                        score=0.80,
                        details=f"Suspicious TLS SNI: {sni}"
                    )
        
        # 4. Detect self-signed certificates (no subject or missing fields)
        if subject == "" or "CN=" not in subject:
            # Potentially self-signed certificate
            return DetectionResult(
                is_detected=True,
                alert_type='tls_attack',
                count=1,
                time_span=0.1,
                score=0.70,
                details=f"Possible self-signed certificate"
            )
        
        return DetectionResult()
    
    def process_event(self, event):
        """Process any event and determine if it should trigger an alert."""
        # Skip if the event is missing key information
        if not hasattr(event, 'saddr') or not hasattr(event, 'daddr'):
            return False
        
        # Extract event type
        app_proto = getattr(event, 'appproto', '')
        event_type = getattr(event, 'type_', '')
        
        # Choose the appropriate analyzer based on protocol/event type
        result = DetectionResult()
        
        if app_proto == 'http' or event_type == 'http':
            result = self.analyze_http_event(event)
        elif app_proto == 'dns' or event_type == 'dns':
            result = self.analyze_dns_event(event)
        elif app_proto == 'ssh' or event_type == 'ssh':
            result = self.analyze_ssh_event(event)
        elif app_proto == 'tls' or event_type == 'tls' or app_proto == 'ssl':
            result = self.analyze_tls_event(event)
        
        # Generate alert if detection occurred
        if result.is_detected and self.alert_callback:
            self._generate_alert(event, result)
            return True
        
        return False
    
    def _generate_alert(self, event, result):
        """Generate an alert based on detection result."""
        # Get basic event information
        src_ip = event.saddr
        dst_ip = event.daddr
        dst_port = getattr(event, 'dport', '0')
        proto = getattr(event, 'proto', '')
        app_proto = getattr(event, 'appproto', '')
        
        # Rate limit alerts (one per source IP per alert type per 30 seconds)
        current_time = time.time()
        alert_key = f"{src_ip}:{result.alert_type}"
        
        if alert_key in self._last_alert_time:
            last_alert = self._last_alert_time[alert_key]
            if current_time - last_alert < 30:
                return  # Don't alert yet, rate limiting
        
        # Update last alert time
        self._last_alert_time[alert_key] = current_time
        
        # Prepare alert data
        alert_data = {
            'flow_id': getattr(event, 'uid', f"immediate-{src_ip}-{dst_ip}-{current_time}"),
            'timestamp': current_time,
            'src_ip': src_ip,
            'src_port': getattr(event, 'sport', ''),
            'dst_ip': dst_ip,
            'dst_port': dst_port,
            'proto': proto,
            'app_proto': app_proto,
            'duration': result.time_span,
            'total_bytes': 0,  # We don't have this info in early detection
            'total_packets': 0,
            'ml_result': {'is_anomalous': True, 'score': result.score},
            'stat_result': {
                'is_anomalous': True,
                'score': result.score,
                'details': [{
                    'feature': result.alert_type,
                    'value': result.count,
                    'z_score': 5.0,  # Placeholder high score
                    'is_outlier': True
                }]
            },
            'combined_score': result.score,
            'is_anomalous': True,
            'early_detection': True,
            'alert_type': result.alert_type,
            'event_count': result.count,
            'rate': result.count / max(result.time_span, 1),
            'details': result.details
        }
        
        # Log the alert
        logger.warning(f"EARLY DETECTION: {result.details}")
        
        # Call the alert handler
        self.alert_callback(alert_data)
    
    def cleanup(self):
        """Clean up old tracking data."""
        current_time = time.time()
        timeout = 600  # 10 minutes
        
        # Clean HTTP tracking
        ip_to_remove = []
        for ip, stats in self._http_tracking['ip_requests'].items():
            if current_time - stats['last_seen'] > timeout:
                ip_to_remove.append(ip)
        
        for ip in ip_to_remove:
            del self._http_tracking['ip_requests'][ip]
        
        # Clean DNS tracking
        ip_to_remove = []
        for ip, stats in self._dns_tracking['ip_queries'].items():
            if current_time - stats['last_seen'] > timeout:
                ip_to_remove.append(ip)
        
        for ip in ip_to_remove:
            del self._dns_tracking['ip_queries'][ip]
        
        domain_to_remove = []
        for domain, stats in self._dns_tracking['domain_requests'].items():
            if current_time - stats['first_seen'] > timeout:
                domain_to_remove.append(domain)
        
        for domain in domain_to_remove:
            del self._dns_tracking['domain_requests'][domain]
        
        # Clean SSH tracking
        key_to_remove = []
        for key, stats in self._ssh_tracking['connection_attempts'].items():
            if current_time - stats['last_seen'] > timeout:
                key_to_remove.append(key)
        
        for key in key_to_remove:
            del self._ssh_tracking['connection_attempts'][key]
        
        # Clean TLS tracking
        ip_to_remove = []
        for ip, stats in self._tls_tracking['connection_attempts'].items():
            if current_time - stats['last_seen'] > timeout:
                ip_to_remove.append(ip)
        
        for ip in ip_to_remove:
            del self._tls_tracking['connection_attempts'][ip]
        
        # Clean alert rate limiting
        alert_to_remove = []
        for alert_key, last_time in self._last_alert_time.items():
            if current_time - last_time > 1800:  # 30 minutes
                alert_to_remove.append(alert_key)
        
        for alert_key in alert_to_remove:
            del self._last_alert_time[alert_key]