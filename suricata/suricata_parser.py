import json
import datetime
from typing import Dict, Any, Union
from suricata.suricata_flows import (
    SuricataFlow,
    SuricataHTTP,
    SuricataDNS,
    SuricataTLS,
    SuricataFile,
    SuricataSSH,
)

class SuricataParser:
    """Parser for Suricata JSON logs"""
    
    def __init__(self):
        self.flow = None

    def get_answers(self, line: dict) -> list:
        """Extract DNS answers from a Suricata DNS event"""
        line = line.get("dns", False)
        if not line:
            return []

        answers: dict = line.get("grouped", False)
        if not answers:
            return []

        cnames: list = answers.get("CNAME", [])
        ips: list = answers.get("A", [])

        return cnames + ips

    def convert_to_datetime(self, timestamp_str):
        """Convert Suricata timestamp to datetime object"""
        try:
            return datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f%z")
        except ValueError:
            try:
                return datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S%z")
            except ValueError:
                return None

    def convert_format(self, timestamp, format_type="iso"):
        """Convert timestamp to desired format"""
        if timestamp is None:
            return None
        
        if format_type == "unixtimestamp":
            if isinstance(timestamp, (int, float)):
                dt = datetime.datetime.fromtimestamp(timestamp)
                return dt.isoformat()
            return timestamp
        
        return timestamp
        
    def process_line(self, line) -> Union[Dict[str, Any], bool]:
        """Process a single line from Suricata eve.json and convert to a flow object"""
        # Convert to dict if it's not a dict already
        if isinstance(line, str):
            try:
                line = json.loads(line)
            except json.JSONDecodeError:
                return False
        
        if not line:
            return False

        # Common fields in all Suricata lines
        event_type = line.get("event_type")
        if not event_type:
            return False
            
        flow_id = line.get("flow_id")
        saddr = line.get("src_ip")
        sport = line.get("src_port")
        daddr = line.get("dest_ip")
        dport = line.get("dest_port")
        proto = line.get("proto")
        appproto = line.get("app_proto", False)

        try:
            timestamp = self.convert_to_datetime(line.get("timestamp"))
            if timestamp:
                timestamp = timestamp.isoformat()
        except Exception:
            timestamp = None

        def get_value_at(field, subfield, default_=False):
            try:
                val = line[field][subfield]
                return val or default_
            except (IndexError, KeyError):
                return default_

        # Process different event types
        if event_type == "flow":
            starttime = self.convert_format(
                get_value_at("flow", "start"), "unixtimestamp"
            )
            endtime = self.convert_format(
                get_value_at("flow", "end"), "unixtimestamp"
            )
            self.flow = SuricataFlow(
                flow_id,
                saddr,
                sport,
                daddr,
                dport,
                proto,
                appproto,
                starttime or timestamp,
                endtime or timestamp,
                int(get_value_at("flow", "pkts_toserver", 0)),
                int(get_value_at("flow", "pkts_toclient", 0)),
                int(get_value_at("flow", "bytes_toserver", 0)),
                int(get_value_at("flow", "bytes_toclient", 0)),
                get_value_at("flow", "state", ""),
            )

        elif event_type == "http":
            self.flow = SuricataHTTP(
                timestamp,
                flow_id,
                saddr,
                sport,
                daddr,
                dport,
                proto,
                appproto,
                get_value_at("http", "http_method", ""),
                get_value_at("http", "hostname", ""),
                get_value_at("http", "url", ""),
                get_value_at("http", "http_user_agent", ""),
                get_value_at("http", "status", ""),
                get_value_at("http", "protocol", ""),
                int(get_value_at("http", "request_body_len", 0)),
                int(get_value_at("http", "length", 0)),
            )

        elif event_type == "dns":
            answers = self.get_answers(line)
            self.flow = SuricataDNS(
                timestamp,
                flow_id,
                saddr,
                sport,
                daddr,
                dport,
                proto,
                appproto,
                get_value_at("dns", "rrname", ""),
                get_value_at("dns", "ttl", ""),
                get_value_at("dns", "rrtype", ""),
                answers,
            )

        elif event_type == "tls":
            self.flow = SuricataTLS(
                timestamp,
                flow_id,
                saddr,
                sport,
                daddr,
                dport,
                proto,
                appproto,
                get_value_at("tls", "version", ""),
                get_value_at("tls", "subject", ""),
                get_value_at("tls", "issuerdn", ""),
                get_value_at("tls", "sni", ""),
                get_value_at("tls", "notbefore", ""),
                get_value_at("tls", "notafter", ""),
            )

        elif event_type == "fileinfo":
            self.flow = SuricataFile(
                timestamp,
                flow_id,
                saddr,
                sport,
                daddr,
                dport,
                proto,
                appproto,
                get_value_at("fileinfo", "size", 0),
            )
            
        elif event_type == "ssh":
            self.flow = SuricataSSH(
                timestamp,
                flow_id,
                saddr,
                sport,
                daddr,
                dport,
                proto,
                appproto,
                get_value_at("ssh", "client", {}).get("software_version", ""),
                get_value_at("ssh", "client", {}).get("proto_version", ""),
                get_value_at("ssh", "server", {}).get("software_version", ""),
            )
        else:
            return False
            
        return self.flow