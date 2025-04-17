#!/usr/bin/env python3
"""
Service Whitelist for Hybrid NIDS
Provides a whitelist of legitimate services to prevent false positives
"""

import logging
import socket
from typing import List, Dict, Tuple, Set, Optional

# Setup logging
logger = logging.getLogger('hybrid-nids')

class ServiceWhitelist:
    """
    Maintains a list of well-known services that should not be flagged as anomalous
    """
    
    def __init__(self):
        """Initialize the whitelist with common legitimate services"""
        # Well-known DNS servers
        self.dns_servers: List[str] = [
            "8.8.8.8",      # Google DNS
            "8.8.4.4",      # Google DNS
            "1.1.1.1",      # Cloudflare DNS
            "1.0.0.1",      # Cloudflare DNS
            "9.9.9.9",      # Quad9 DNS
            "149.112.112.112", # Quad9 DNS
            "208.67.222.222",  # OpenDNS
            "208.67.220.220",  # OpenDNS
            "64.6.64.6",       # Verisign DNS
            "64.6.65.6",       # Verisign DNS
            # Add your local DNS server if needed
        ]
        
        # Common NTP servers
        self.ntp_servers: List[str] = [
            "pool.ntp.org",
            "time.google.com",
            "time.windows.com",
            "time.apple.com",
            "time.nist.gov"
        ]
        
        # Resolve domain names to IPs
        self.ntp_server_ips: Set[str] = self._resolve_domains(self.ntp_servers)
        
        # Common update/CDN services
        self.update_services: Dict[str, List[int]] = {
            # Format: "ip": [port1, port2, ...]
            "23.221.50.32": [80, 443],    # Akamai
            "104.16.0.0/12": [80, 443],   # Cloudflare range
            "151.101.0.0/16": [80, 443],  # Fastly range
        }
        
        # Other well-known services
        self.known_services: Dict[Tuple[str, int], str] = {
            # Format: (ip, port): "Service Name"
            ("13.107.21.200", 443): "Microsoft Services",
            ("13.107.42.12", 443): "Microsoft Services",
            # Add more as needed
        }
        
        # Common broadcast/multicast addresses
        self.multicast_addresses: List[str] = [
            "224.0.0.0/24",  # Local multicast
            "239.255.255.250", # SSDP
            "224.0.0.251",   # mDNS
            "255.255.255.255" # Broadcast
        ]
        
        # Add pfSense internal interfaces to trusted list
        self.pfsense_interfaces = set()
        self.pfsense_interfaces.add("10.0.0.2")  # Add your pfSense interface IP
        
        logger.info(f"Added {len(self.pfsense_interfaces)} pfSense interfaces to trusted list")
        
        logger.info(f"Initialized service whitelist with {len(self.dns_servers)} DNS servers, "
                   f"{len(self.ntp_server_ips)} NTP servers, and {len(self.pfsense_interfaces)} Pfsense approved IPs")
    
    def _resolve_domains(self, domains: List[str]) -> Set[str]:
        """Resolve domain names to IP addresses"""
        resolved_ips = set()
        
        for domain in domains:
            try:
                # Get all possible IPs for the domain
                ips = socket.gethostbyname_ex(domain)[2]
                resolved_ips.update(ips)
                logger.debug(f"Resolved {domain} to {', '.join(ips)}")
            except socket.gaierror:
                logger.warning(f"Could not resolve domain: {domain}")
        
        return resolved_ips
    
    def _is_in_cidr(self, ip: str, cidr: str) -> bool:
        """Check if an IP is in a CIDR range"""
        if '/' not in cidr:
            return ip == cidr
            
        try:
            from ipaddress import ip_network, ip_address
            return ip_address(ip) in ip_network(cidr)
        except (ValueError, ImportError):
            # Fallback if ipaddress module is not available
            network, bits = cidr.split('/')
            bits = int(bits)
            
            # Convert IP to integer
            def ip_to_int(ip_str):
                octets = ip_str.split('.')
                return sum(int(octet) << (8 * (3 - i)) for i, octet in enumerate(octets))
            
            ip_int = ip_to_int(ip)
            network_int = ip_to_int(network)
            
            # Create mask and check if IPs match under the mask
            mask = (1 << 32) - (1 << (32 - bits))
            return (ip_int & mask) == (network_int & mask)
    
    def is_whitelisted(self, ip_addr: str, port: int, proto: str = None) -> bool:
        """
        Check if a destination IP and port combination is whitelisted
        
        Args:
            ip_addr: Destination IP address
            port: Destination port
            proto: Protocol (optional)
            
        Returns:
            True if the connection is to a whitelisted service
        """
        
        # Check pfSense IP
        if ip_addr in self.pfsense_interfaces:
            logger.debug(f"Whitelisted PfSense Interface: {ip_addr}:{port}")
            return True
        
        # Check DNS servers (port 53)
        if port == 53 and ip_addr in self.dns_servers:
            logger.debug(f"Whitelisted DNS server: {ip_addr}:{port}")
            return True
            
        # Check NTP servers (port 123)
        if port == 123 and (proto == "UDP" or proto == "TCP"):
            if ip_addr in self.ntp_server_ips:
                logger.debug(f"Whitelisted NTP server: {ip_addr}:{port}")
                return True
            
        # Check other known services
        if (ip_addr, port) in self.known_services:
            service_name = self.known_services[(ip_addr, port)]
            logger.debug(f"Whitelisted service: {ip_addr}:{port} ({service_name})")
            return True
            
        # Check update/CDN services
        for cidr, ports in self.update_services.items():
            if port in ports and self._is_in_cidr(ip_addr, cidr):
                logger.debug(f"Whitelisted CDN/update service: {ip_addr}:{port}")
                return True
        
        # Check multicast/broadcast
        for addr in self.multicast_addresses:
            if self._is_in_cidr(ip_addr, addr):
                logger.debug(f"Whitelisted multicast/broadcast: {ip_addr}:{port}")
                return True
            
        return False
        
    def add_dns_server(self, ip_addr: str) -> None:
        """Add a DNS server to the whitelist"""
        if ip_addr not in self.dns_servers:
            self.dns_servers.append(ip_addr)
            logger.info(f"Added DNS server to whitelist: {ip_addr}")
    
    def add_known_service(self, ip_addr: str, port: int, service_name: str) -> None:
        """Add a known service to the whitelist"""
        if (ip_addr, port) not in self.known_services:
            self.known_services[(ip_addr, port)] = service_name
            logger.info(f"Added service to whitelist: {ip_addr}:{port} ({service_name})")