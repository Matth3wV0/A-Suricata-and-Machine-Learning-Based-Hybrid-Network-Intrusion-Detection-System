#!/usr/bin/env python3
"""
Session Manager Module for Hybrid NIDS
Aggregates Suricata events by flow_id and maintains session state
"""

import time
import logging
import datetime
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# Import Suricata flow types
from utils.suricata_flows import (
    SuricataFlow,
    SuricataHTTP,
    SuricataDNS,
    SuricataTLS,
    SuricataFile,
    SuricataSSH,
)

# Setup logging
logger = logging.getLogger('hybrid-nids')

@dataclass
class SuricataSession:
    """Enriched session dataclass that combines multiple Suricata events"""
    flow_id: str
    saddr: str
    sport: str
    daddr: str
    dport: str
    proto: str
    
    # Flow metrics
    starttime: str = None
    endtime: str = None
    duration: float = 0
    state: str = ""
    appproto: str = ""
    
    # Packet/byte counters
    total_fwd_packets: int = 0
    total_bwd_packets: int = 0
    total_fwd_bytes: int = 0
    total_bwd_bytes: int = 0
    
    # HTTP data
    http_methods: List[str] = field(default_factory=list)
    http_status_codes: List[str] = field(default_factory=list)
    http_hosts: List[str] = field(default_factory=list)
    http_uris: List[str] = field(default_factory=list)
    http_user_agents: List[str] = field(default_factory=list)
    
    # DNS data
    dns_queries: List[str] = field(default_factory=list)
    dns_answers: List[str] = field(default_factory=list)
    dns_rrtypes: List[str] = field(default_factory=list)
    
    # TLS data
    tls_sni: List[str] = field(default_factory=list)
    tls_versions: List[str] = field(default_factory=list)
    tls_subjects: List[str] = field(default_factory=list)
    tls_issuers: List[str] = field(default_factory=list)
    
    # SSH data
    ssh_client_versions: List[str] = field(default_factory=list)
    ssh_server_versions: List[str] = field(default_factory=list)
    
    # Event counts
    http_event_count: int = 0
    dns_event_count: int = 0
    tls_event_count: int = 0
    ssh_event_count: int = 0
    file_event_count: int = 0
    
    # Derived metrics (calculated during finalization)
    flow_bytes_per_sec: float = 0
    flow_packets_per_sec: float = 0
    down_up_ratio: float = 0
    
    # Status flags
    is_complete: bool = False
    has_app_data: bool = False
    last_updated: float = field(default_factory=time.time)
    
    def update_from_flow(self, flow: SuricataFlow) -> None:
        """Update session with flow information"""
        if not self.starttime or flow.starttime < self.starttime:
            self.starttime = flow.starttime
        
        if not self.endtime or flow.endtime > self.endtime:
            self.endtime = flow.endtime
        
        self.state = flow.state
        self.appproto = flow.appproto if flow.appproto else self.appproto
        
        # Update packet and byte counts
        self.total_fwd_packets = flow.spkts
        self.total_bwd_packets = flow.dpkts
        self.total_fwd_bytes = flow.sbytes
        self.total_bwd_bytes = flow.dbytes
        
        # Update duration if available
        if flow.dur:
            self.duration = float(flow.dur)
            
        self.last_updated = time.time()
    
    def update_from_http(self, http: SuricataHTTP) -> None:
        """Update session with HTTP information"""
        self.has_app_data = True
        self.http_event_count += 1
        
        # Update basic flow properties if not already set
        if not self.starttime:
            self.starttime = http.starttime
        
        self.appproto = "http" if not self.appproto else self.appproto
        
        # Append HTTP details
        if http.method and http.method not in self.http_methods:
            self.http_methods.append(http.method)
            
        if http.status_code and http.status_code not in self.http_status_codes:
            self.http_status_codes.append(http.status_code)
            
        if http.host and http.host not in self.http_hosts:
            self.http_hosts.append(http.host)
            
        if http.uri and http.uri not in self.http_uris:
            self.http_uris.append(http.uri)
            
        if http.user_agent and http.user_agent not in self.http_user_agents:
            self.http_user_agents.append(http.user_agent)
            
        self.last_updated = time.time()
    
    def update_from_dns(self, dns: SuricataDNS) -> None:
        """Update session with DNS information"""
        self.has_app_data = True
        self.dns_event_count += 1
        
        # Update basic flow properties if not already set
        if not self.starttime:
            self.starttime = dns.starttime
        
        self.appproto = "dns" if not self.appproto else self.appproto
        
        # Append DNS details
        if dns.query and dns.query not in self.dns_queries:
            self.dns_queries.append(dns.query)
            
        if dns.qtype_name and dns.qtype_name not in self.dns_rrtypes:
            self.dns_rrtypes.append(dns.qtype_name)
            
        # Process DNS answers
        if dns.answers:
            for answer in dns.answers:
                if isinstance(answer, str) and answer not in self.dns_answers:
                    self.dns_answers.append(answer)
                elif isinstance(answer, dict) and answer.get('value') and answer.get('value') not in self.dns_answers:
                    self.dns_answers.append(answer.get('value'))
                    
        self.last_updated = time.time()
    
    def update_from_tls(self, tls: SuricataTLS) -> None:
        """Update session with TLS information"""
        self.has_app_data = True
        self.tls_event_count += 1
        
        # Update basic flow properties if not already set
        if not self.starttime:
            self.starttime = tls.starttime
        
        self.appproto = "tls" if not self.appproto else self.appproto
        
        # Append TLS details
        if tls.server_name and tls.server_name not in self.tls_sni:
            self.tls_sni.append(tls.server_name)
            
        if tls.sslversion and tls.sslversion not in self.tls_versions:
            self.tls_versions.append(tls.sslversion)
            
        if tls.subject and tls.subject not in self.tls_subjects:
            self.tls_subjects.append(tls.subject)
            
        if tls.issuer and tls.issuer not in self.tls_issuers:
            self.tls_issuers.append(tls.issuer)
            
        self.last_updated = time.time()
    
    def update_from_ssh(self, ssh: SuricataSSH) -> None:
        """Update session with SSH information"""
        self.has_app_data = True
        self.ssh_event_count += 1
        
        # Update basic flow properties if not already set
        if not self.starttime:
            self.starttime = ssh.starttime
        
        self.appproto = "ssh" if not self.appproto else self.appproto
        
        # Append SSH details
        if ssh.client and ssh.client not in self.ssh_client_versions:
            self.ssh_client_versions.append(ssh.client)
            
        if ssh.server and ssh.server not in self.ssh_server_versions:
            self.ssh_server_versions.append(ssh.server)
            
        self.last_updated = time.time()
    
    def update_from_file(self, file: SuricataFile) -> None:
        """Update session with file information"""
        self.has_app_data = True
        self.file_event_count += 1
        
        # Update basic flow properties if not already set
        if not self.starttime:
            self.starttime = file.starttime
            
        self.last_updated = time.time()
    
    def finalize(self) -> None:
        """Finalize session by calculating derived metrics"""
        # Calculate duration if not already set
        if self.duration == 0 and self.starttime and self.endtime:
            try:
                start = datetime.datetime.fromisoformat(self.starttime.replace('Z', '+00:00'))
                end = datetime.datetime.fromisoformat(self.endtime.replace('Z', '+00:00'))
                self.duration = (end - start).total_seconds()
            except Exception as e:
                logger.warning(f"Failed to calculate duration for flow {self.flow_id}: {e}")
        
        # Ensure duration is at least 0.001 to avoid division by zero
        safe_duration = max(self.duration, 0.001)
        
        # Calculate bytes per second
        total_bytes = self.total_fwd_bytes + self.total_bwd_bytes
        self.flow_bytes_per_sec = total_bytes / safe_duration
        
        # Calculate packets per second
        total_packets = self.total_fwd_packets + self.total_bwd_packets
        self.flow_packets_per_sec = total_packets / safe_duration
        
        # Calculate down/up ratio
        if self.total_fwd_bytes > 0:
            self.down_up_ratio = self.total_bwd_bytes / self.total_fwd_bytes
        else:
            self.down_up_ratio = 0
        
        self.is_complete = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return asdict(self)
    
    def get_app_layer_info(self) -> Dict[str, Any]:
        """Get application layer information for enriched features"""
        app_info = {
            # HTTP info
            'http_event_count': self.http_event_count,
            'unique_http_methods': len(self.http_methods),
            'has_http_error': any(code.startswith('4') or code.startswith('5') for code in self.http_status_codes),
            'has_http_auth': '401' in self.http_status_codes or '407' in self.http_status_codes,
            
            # DNS info
            'dns_event_count': self.dns_event_count,
            'unique_dns_queries': len(self.dns_queries),
            'dns_answer_count': len(self.dns_answers),
            
            # TLS info
            'tls_event_count': self.tls_event_count,
            'tls_sni_count': len(self.tls_sni),
            
            # SSH info
            'ssh_event_count': self.ssh_event_count,
            
            # File info
            'file_event_count': self.file_event_count,
            
            # Mixed app data
            'has_mixed_app_data': len([x for x in [self.http_event_count, self.dns_event_count, 
                                                   self.tls_event_count, self.ssh_event_count] 
                                      if x > 0]) > 1
        }
        
        return app_info


class SessionManager:
    """Manages flow sessions and aggregates related events"""
    
    def __init__(self, session_timeout: int = 60, max_sessions: int = 10000):
        """Initialize session manager
        
        Args:
            session_timeout: Time in seconds before a session is considered expired
            max_sessions: Maximum number of sessions to keep in memory
        """
        self.session_timeout = session_timeout
        self.max_sessions = max_sessions
        self.sessions: Dict[str, SuricataSession] = {}
        self.closed_sessions: List[SuricataSession] = []
        self.session_by_ip: Dict[str, Set[str]] = defaultdict(set)
        self.session_by_dest_port: Dict[int, Set[str]] = defaultdict(set)
        
        # Statistics
        self.stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'closed_sessions': 0,
            'expired_sessions': 0,
            'flow_events': 0,
            'http_events': 0,
            'dns_events': 0,
            'tls_events': 0,
            'ssh_events': 0,
            'file_events': 0
        }
    
    def process_event(self, event: Any) -> Optional[SuricataSession]:
        """Process Suricata event and update corresponding session
        
        Args:
            event: Suricata event object
            
        Returns:
            Session if it was finalized by this event, None otherwise
        """
        if not event or not hasattr(event, 'uid'):
            return None
        
        flow_id = event.uid
        finalized_session = None
        
        # Check if session exists, create if not
        if flow_id not in self.sessions:
            # Create new session if it's a flow or an app-layer event
            if isinstance(event, (SuricataFlow, SuricataHTTP, SuricataDNS, SuricataTLS, SuricataSSH, SuricataFile)):
                self.sessions[flow_id] = SuricataSession(
                    flow_id=flow_id,
                    saddr=event.saddr,
                    sport=event.sport,
                    daddr=event.daddr,
                    dport=event.dport,
                    proto=event.proto
                )
                self.stats['total_sessions'] += 1
                self.stats['active_sessions'] += 1
                
                # Add to IP and port indexes
                self.session_by_ip[event.saddr].add(flow_id)
                try:
                    port = int(event.dport)
                    self.session_by_dest_port[port].add(flow_id)
                except (ValueError, TypeError):
                    pass
        
        # Update session based on event type
        if flow_id in self.sessions:
            session = self.sessions[flow_id]
            
            if isinstance(event, SuricataFlow):
                session.update_from_flow(event)
                self.stats['flow_events'] += 1
                
                # Check if flow is complete
                if event.state in ['closed', 'established', 'fin', 'rst']:
                    session.finalize()
                    finalized_session = session
                    self._close_session(flow_id)
                    
            elif isinstance(event, SuricataHTTP):
                session.update_from_http(event)
                self.stats['http_events'] += 1
                
            elif isinstance(event, SuricataDNS):
                session.update_from_dns(event)
                self.stats['dns_events'] += 1
                
            elif isinstance(event, SuricataTLS):
                session.update_from_tls(event)
                self.stats['tls_events'] += 1
                
            elif isinstance(event, SuricataSSH):
                session.update_from_ssh(event)
                self.stats['ssh_events'] += 1
                
            elif isinstance(event, SuricataFile):
                session.update_from_file(event)
                self.stats['file_events'] += 1
        
        return finalized_session
    
    def get_session(self, flow_id: str) -> Optional[SuricataSession]:
        """Get session by flow_id"""
        return self.sessions.get(flow_id)
    
    def get_sessions_by_ip(self, ip_addr: str) -> List[SuricataSession]:
        """Get all sessions for a specific IP address"""
        flow_ids = self.session_by_ip.get(ip_addr, set())
        return [self.sessions[fid] for fid in flow_ids if fid in self.sessions]
    
    def get_sessions_by_dest_port(self, port: int) -> List[SuricataSession]:
        """Get all sessions for a specific destination port"""
        flow_ids = self.session_by_dest_port.get(port, set())
        return [self.sessions[fid] for fid in flow_ids if fid in self.sessions]
    
    def cleanup_expired_sessions(self) -> List[SuricataSession]:
        """Cleanup expired sessions
        
        Returns:
            List of expired sessions that were finalized
        """
        current_time = time.time()
        expired_flow_ids = []
        finalized_sessions = []
        
        # Find expired sessions
        for flow_id, session in self.sessions.items():
            if current_time - session.last_updated > self.session_timeout:
                expired_flow_ids.append(flow_id)
        
        # Close expired sessions
        for flow_id in expired_flow_ids:
            session = self.sessions[flow_id]
            session.finalize()
            finalized_sessions.append(session)
            self._close_session(flow_id)
            self.stats['expired_sessions'] += 1
        
        # Enforce max sessions limit if needed
        if len(self.sessions) > self.max_sessions:
            # Get oldest sessions
            oldest_sessions = sorted(
                self.sessions.items(), 
                key=lambda x: x[1].last_updated
            )[:len(self.sessions) - self.max_sessions]
            
            # Close oldest sessions
            for flow_id, session in oldest_sessions:
                session.finalize()
                finalized_sessions.append(session)
                self._close_session(flow_id)
        
        return finalized_sessions
    
    def _close_session(self, flow_id: str) -> None:
        """Close a session and move it to closed_sessions"""
        if flow_id in self.sessions:
            session = self.sessions[flow_id]
            
            # Make sure it's finalized
            if not session.is_complete:
                session.finalize()
            
            # Move to closed sessions
            self.closed_sessions.append(session)
            
            # Remove from active sessions
            del self.sessions[flow_id]
            
            # Remove from indexes
            self.session_by_ip[session.saddr].discard(flow_id)
            try:
                port = int(session.dport)
                self.session_by_dest_port[port].discard(flow_id)
            except (ValueError, TypeError):
                pass
            
            # Update stats
            self.stats['active_sessions'] -= 1
            self.stats['closed_sessions'] += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get session manager statistics"""
        return self.stats.copy()
