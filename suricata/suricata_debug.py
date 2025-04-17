#!/usr/bin/env python3
"""
Debug script to analyze Suricata eve.json files and print out their structure
"""

import json
import argparse
import os
import sys
from collections import defaultdict

def analyze_eve_json(file_path):
    """Analyze the structure of eve.json entries"""
    print(f"Analyzing file: {file_path}")
    
    # Counters for statistics
    total_entries = 0
    event_types = defaultdict(int)
    fields_by_type = defaultdict(set)
    flow_fields = set()
    
    # Process file line by line
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                total_entries += 1
                
                # Parse JSON
                entry = json.loads(line)
                
                # Get event type
                event_type = entry.get('event_type', 'unknown')
                event_types[event_type] += 1
                
                # Record fields for this event type
                fields_by_type[event_type].update(entry.keys())
                
                # If this is a flow event, record all flow fields
                if event_type == 'flow' and 'flow' in entry:
                    flow_fields.update(entry['flow'].keys())
                
                # For the first 10 entries of each type, print details
                if event_types[event_type] <= 3:
                    print(f"\n--- {event_type} Entry #{event_types[event_type]} (Line {line_num}) ---")
                    print(json.dumps(entry, indent=2)[:1000])  # Truncate to 1000 chars
                
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON at line {line_num}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Total entries: {total_entries}")
    print("\nEvent Types:")
    for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {event_type}: {count} entries")
    
    # Print fields for flow events
    if 'flow' in event_types:
        print("\nFlow Event Fields:")
        for field in sorted(fields_by_type['flow']):
            print(f"  {field}")
        
        print("\nFlow Object Fields:")
        for field in sorted(flow_fields):
            print(f"  flow.{field}")
    
    # Print fields for http events
    if 'http' in event_types:
        print("\nHTTP Event Fields:")
        for field in sorted(fields_by_type['http']):
            print(f"  {field}")
            
    # Recommend features to extract
    print("\n=== RECOMMENDED FEATURE MAPPING ===")
    print("Based on the file structure, here's how to map CICIDS2017 features:")
    
    if 'flow' in event_types:
        flow_entry = None
        # Find a flow entry to analyze
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('event_type') == 'flow':
                        flow_entry = entry
                        break
                except:
                    continue
        
        if flow_entry:
            # 1. Destination Port
            if 'dest_port' in flow_entry:
                print(f"✓ Destination Port → flow_dict['dest_port'] or flow_dict['dport']")
            else:
                print(f"✗ Destination Port - Field not found")
            
            # 2. Flow Duration
            if 'flow' in flow_entry and 'start' in flow_entry['flow'] and 'end' in flow_entry['flow']:
                print(f"✓ Flow Duration → Compute from flow_dict['flow']['start'] and flow_dict['flow']['end']")
            elif 'dur' in flow_entry:
                print(f"✓ Flow Duration → flow_dict['dur']")
            else:
                print(f"✗ Flow Duration - Fields not found")
            
            # 3. Total Fwd Packets
            if 'flow' in flow_entry and 'pkts_toserver' in flow_entry['flow']:
                print(f"✓ Total Fwd Packets → flow_dict['flow']['pkts_toserver']")
            elif 'spkts' in flow_entry:
                print(f"✓ Total Fwd Packets → flow_dict['spkts']")
            else:
                print(f"✗ Total Fwd Packets - Field not found")
            
            # 4. Total Backward Packets
            if 'flow' in flow_entry and 'pkts_toclient' in flow_entry['flow']:
                print(f"✓ Total Backward Packets → flow_dict['flow']['pkts_toclient']")
            elif 'dpkts' in flow_entry:
                print(f"✓ Total Backward Packets → flow_dict['dpkts']")
            else:
                print(f"✗ Total Backward Packets - Field not found")
            
            # 5. Total Length of Fwd Packets
            if 'flow' in flow_entry and 'bytes_toserver' in flow_entry['flow']:
                print(f"✓ Total Length of Fwd Packets → flow_dict['flow']['bytes_toserver']")
            elif 'sbytes' in flow_entry:
                print(f"✓ Total Length of Fwd Packets → flow_dict['sbytes']")
            else:
                print(f"✗ Total Length of Fwd Packets - Field not found")
            
            # 6. Total Length of Bwd Packets
            if 'flow' in flow_entry and 'bytes_toclient' in flow_entry['flow']:
                print(f"✓ Total Length of Bwd Packets → flow_dict['flow']['bytes_toclient']")
            elif 'dbytes' in flow_entry:
                print(f"✓ Total Length of Bwd Packets → flow_dict['dbytes']")
            else:
                print(f"✗ Total Length of Bwd Packets - Field not found")

def main():
    parser = argparse.ArgumentParser(description='Debug Suricata eve.json structure')
    parser.add_argument('file_path', help='Path to eve.json file')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.file_path):
        print(f"Error: File {args.file_path} does not exist")
        sys.exit(1)
    
    analyze_eve_json(args.file_path)

if __name__ == "__main__":
    main()