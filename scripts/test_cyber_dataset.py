"""
Test script to demonstrate CyberLogDataset features and structure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from acie.dataset import CyberLogDataset

def analyze_log_features(log, dataset):
    """Analyze and display the structured features of a cyber log"""
    offset = 0
    
    print("\n" + "="*60)
    print("CYBER LOG FEATURE BREAKDOWN")
    print("="*60)
    
    # Timestamp
    timestamp = log[offset:offset+dataset.field_dims['timestamp']]
    offset += dataset.field_dims['timestamp']
    print(f"\n1. TIMESTAMP ({dataset.field_dims['timestamp']} dims)")
    print(f"   High activity indicator: {timestamp[0]:.3f}")
    print(f"   Time interval: {timestamp[1]:.3f}")
    print(f"   Off-hours indicator: {timestamp[2]:.3f}")
    
    # Source IP
    source_ip = log[offset:offset+dataset.field_dims['source_ip']]
    offset += dataset.field_dims['source_ip']
    print(f"\n2. SOURCE IP ({dataset.field_dims['source_ip']} dims)")
    print(f"   Internal IP flag: {source_ip[0]:.3f}")
    print(f"   IP encoding: [{source_ip[1]:.3f}, {source_ip[2]:.3f}, {source_ip[3]:.3f}, {source_ip[4]:.3f}]")
    print(f"   Compromised flag: {source_ip[10]:.3f}")
    
    # Destination IP
    dest_ip = log[offset:offset+dataset.field_dims['dest_ip']]
    offset += dataset.field_dims['dest_ip']
    print(f"\n3. DESTINATION IP ({dataset.field_dims['dest_ip']} dims)")
    print(f"   Internal IP flag: {dest_ip[0]:.3f}")
    print(f"   IP encoding: [{dest_ip[1]:.3f}, {dest_ip[2]:.3f}, {dest_ip[3]:.3f}, {dest_ip[4]:.3f}]")
    print(f"   Database server flag: {dest_ip[15]:.3f}")
    print(f"   External suspicious: {dest_ip[16]:.3f}")
    
    # Process ID
    process_id = log[offset:offset+dataset.field_dims['process_id']]
    offset += dataset.field_dims['process_id']
    print(f"\n4. PROCESS ID ({dataset.field_dims['process_id']} dims)")
    print(f"   Network service: {process_id[0]:.3f}")
    print(f"   System-level: {process_id[1]:.3f}")
    print(f"   Child process: {process_id[2]:.3f}")
    print(f"   SMB/RDP indicator: {process_id[5]:.3f}")
    
    # Event Type
    event_type = log[offset:offset+dataset.field_dims['event_type']]
    offset += dataset.field_dims['event_type']
    print(f"\n5. EVENT TYPE ({dataset.field_dims['event_type']} dims)")
    
    # Find one-hot encoded event
    base_events = ['PROCESS_CREATE', 'NETWORK_CONNECT', 'FILE_ACCESS',
                   'REGISTRY_MODIFY', 'USER_LOGIN', 'PRIVILEGE_CHANGE',
                   'DATA_TRANSFER', 'SERVICE_START', 'PROCESS_TERMINATE', 'OTHER']
    
    event_idx = torch.argmax(torch.tensor(event_type[:10])).item()
    print(f"   Base Event: {base_events[event_idx]}")
    print(f"   Lateral movement signature: {event_type[20]:.3f}")
    print(f"   Exfiltration signature: {event_type[21]:.3f}")
    print(f"   Privilege escalation signature: {event_type[22]:.3f}")
    print(f"   Malware signature: {event_type[23]:.3f}")
    
    print("="*60)


def main():
    print("="*60)
    print("CYBER LOG DATASET TESTING")
    print("="*60)
    
    # Create dataset
    dataset = CyberLogDataset(num_samples=50, input_dim=100, num_classes=5, seed=42)
    
    print(f"\nDataset created: {len(dataset)} samples")
    print(f"Input dimension: {dataset.input_dim}")
    print(f"\nField dimensions:")
    for field, dim in dataset.field_dims.items():
        print(f"  {field}: {dim} dims")
    
    print("\n" + "="*60)
    print("ATTACK PATTERNS")
    print("="*60)
    for label, pattern in dataset.attack_patterns.items():
        print(f"  {label}: {pattern}")
    
    # Show examples of each attack type
    print("\n" + "="*60)
    print("SAMPLE LOGS FOR EACH ATTACK TYPE")
    print("="*60)
    
    # Find one example of each attack type
    shown_types = set()
    for idx in range(len(dataset)):
        log, label = dataset[idx]
        label_val = label.item()
        
        if label_val not in shown_types:
            print(f"\n{'='*60}")
            print(f"Attack Type {label_val}: {dataset.get_attack_pattern(label_val)}")
            analyze_log_features(log.numpy(), dataset)
            shown_types.add(label_val)
            
            if len(shown_types) == 5:
                break
    
    # Test lateral movement detection
    print("\n" + "="*60)
    print("LATERAL MOVEMENT PATTERN ANALYSIS")
    print("="*60)
    
    lateral_logs = []
    for idx in range(len(dataset)):
        log, label = dataset[idx]
        if label.item() == 1:  # LATERAL_MOVEMENT
            lateral_logs.append(log)
            if len(lateral_logs) >= 3:
                break
    
    if lateral_logs:
        print(f"\nFound {len(lateral_logs)} lateral movement samples")
        print("Analyzing average feature patterns...")
        
        avg_log = torch.stack(lateral_logs).mean(dim=0)
        
        offset = dataset.field_dims['timestamp']
        source_ip = avg_log[offset:offset+dataset.field_dims['source_ip']]
        offset += dataset.field_dims['source_ip']
        dest_ip = avg_log[offset:offset+dataset.field_dims['dest_ip']]
        
        print(f"\nAverage Source IP (Compromised Node A):")
        print(f"  Internal flag: {source_ip[0]:.3f}")
        print(f"  Compromised flag: {source_ip[10]:.3f}")
        
        print(f"\nAverage Destination IP (Database Node B):")
        print(f"  Internal flag: {dest_ip[0]:.3f}")
        print(f"  Database flag: {dest_ip[15]:.3f}")
        
        print("\n[SUCCESS] Strong causal pattern expected: Compromised Node A -> Database Node B")
    
    print("\n" + "="*60)
    print("[SUCCESS] CYBER LOG DATASET TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
