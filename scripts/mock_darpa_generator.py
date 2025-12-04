"""
Generate mock DARPA OpTC dataset in JSON-Lines format.
Creates realistic test data following the "Five Directions" schema.
"""

import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse


class MockDARPAGenerator:
    """Generate realistic DARPA OpTC events for testing"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        
        # Internal network IPs
        self.internal_ips = [
            f"192.168.1.{i}" for i in range(1, 255)
        ] + [
            f"10.0.0.{i}" for i in range(1, 100)
        ]
        
        # External IPs
        self.external_ips = [
            "8.8.8.8",           # Google DNS
            "1.1.1.1",           # Cloudflare DNS
            "93.184.216.34",     # Example.com
            "151.101.1.140",     # Reddit
            "157.240.241.35"     # Facebook
        ]
        
        # Action types
        self.actions = [
            'FLOW_START', 'FLOW_END', 'PROCESS_CREATE', 'PROCESS_TERMINATE',
            'FILE_READ', 'FILE_WRITE', 'FILE_DELETE',
            'REGISTRY_READ', 'REGISTRY_WRITE',
            'USER_LOGIN', 'PRIVILEGE_CHANGE', 'DATA_TRANSFER', 'SERVICE_START'
        ]
        
        # Process IDs
        self.pids = list(range(100, 9999))
        
        # Attack scenarios
        self.scenarios = {
            'normal': 0.70,           # 70% normal traffic
            'lateral_movement': 0.10,  # 10% lateral movement
            'exfiltration': 0.10,      # 10% exfiltration
            'privilege_escalation': 0.05,  # 5% privilege escalation
            'malware': 0.05            # 5% malware execution
        }
        
        # Start time (Nov 1, 2025)
        self.start_time = datetime(2025, 11, 1, 0, 0, 0)
    
    def _generate_timestamp(self, base_time):
        """Generate realistic timestamp in nanoseconds"""
        # Add random offset (0-60 seconds)
        offset = timedelta(seconds=random.uniform(0, 60))
        timestamp = base_time + offset
        
        # Convert to nanoseconds since epoch
        return int(timestamp.timestamp() * 1e9)
    
    def _select_scenario(self):
        """Select attack scenario based on probability distribution"""
        rand = random.random()
        cumulative = 0.0
        
        for scenario, prob in self.scenarios.items():
            cumulative += prob
            if rand <= cumulative:
                return scenario
        
        return 'normal'
    
    def generate_normal_event(self, timestamp):
        """Generate normal network activity"""
        src_ip = random.choice(self.internal_ips)
        dest_ip = random.choice(self.internal_ips)
        action = random.choice(['FILE_READ', 'FILE_WRITE', 'PROCESS_CREATE', 'SERVICE_START'])
        pid = random.choice(self.pids)
        
        return {
            'timestamp': timestamp,
            'action': action,
            'actor': {
                'src_ip': src_ip,
                'pid': pid
            },
            'object': {
                'dest_ip': dest_ip,
                'path': f'/usr/bin/program_{random.randint(1,100)}'
            },
            'properties': {
                'pid': pid,
                'uid': random.randint(1000, 2000),
                'suspicious': False
            }
        }
    
    def generate_lateral_movement_event(self, timestamp):
        """Generate lateral movement attack pattern"""
        # Compromised host -> Database server
        compromised_host = random.choice(self.internal_ips[:50])  # Workstation range
        database_server = random.choice(self.internal_ips[200:220])  # Server range
        
        return {
            'timestamp': timestamp,
            'action': 'FLOW_START',
            'actor': {
                'src_ip': compromised_host,
                'pid': random.choice(self.pids)
            },
            'object': {
                'dest_ip': database_server,
                'port': 3306,  # MySQL
                'protocol': 'TCP'
            },
            'properties': {
                'pid': random.choice(self.pids),
                'uid': 0,  # Root access
                'suspicious': True,
                'bytes_transferred': random.randint(1000, 100000)
            }
        }
    
    def generate_exfiltration_event(self, timestamp):
        """Generate data exfiltration pattern"""
        internal_host = random.choice(self.internal_ips)
        external_ip = random.choice(self.external_ips)
        
        return {
            'timestamp': timestamp,
            'action': 'DATA_TRANSFER',
            'actor': {
                'src_ip': internal_host,
                'pid': random.choice(self.pids)
            },
            'object': {
                'dest_ip': external_ip,
                'port': 443,  # HTTPS
                'protocol': 'TCP'
            },
            'properties': {
                'pid': random.choice(self.pids),
                'uid': random.randint(1000, 2000),
                'suspicious': True,
                'bytes_transferred': random.randint(100000, 10000000)  # Large transfer
            }
        }
    
    def generate_privilege_escalation_event(self, timestamp):
        """Generate privilege escalation pattern"""
        host = random.choice(self.internal_ips)
        
        return {
            'timestamp': timestamp,
            'action': 'PRIVILEGE_CHANGE',
            'actor': {
                'src_ip': host,
                'pid': random.choice(self.pids),
                'uid': random.randint(1000, 2000)  # Normal user
            },
            'object': {
                'uid': 0,  # Escalate to root
                'gid': 0
            },
            'properties': {
                'pid': random.choice(self.pids),
                'uid': 0,  # Now root
                'suspicious': True,
                'method': 'sudo'
            }
        }
    
    def generate_malware_event(self, timestamp):
        """Generate malware execution pattern"""
        host = random.choice(self.internal_ips)
        
        return {
            'timestamp': timestamp,
            'action': 'PROCESS_CREATE',
            'actor': {
                'src_ip': host,
                'pid': random.choice(self.pids),
                'path': '/usr/bin/bash'
            },
            'object': {
                'path': f'/tmp/malware_{random.randint(1000,9999)}',
                'pid': random.choice(self.pids)
            },
            'properties': {
                'pid': random.choice(self.pids),
                'uid': random.randint(1000, 2000),
                'suspicious': True,
                'parent_pid': random.choice(self.pids)
            }
        }
    
    def generate_event(self, timestamp):
        """Generate a single event based on scenario selection"""
        scenario = self._select_scenario()
        
        if scenario == 'normal':
            return self.generate_normal_event(timestamp)
        elif scenario == 'lateral_movement':
            return self.generate_lateral_movement_event(timestamp)
        elif scenario == 'exfiltration':
            return self.generate_exfiltration_event(timestamp)
        elif scenario == 'privilege_escalation':
            return self.generate_privilege_escalation_event(timestamp)
        elif scenario == 'malware':
            return self.generate_malware_event(timestamp)
        
        return self.generate_normal_event(timestamp)
    
    def generate_dataset(self, num_events, output_path):
        """Generate complete dataset and save to JSONL"""
        print(f"Generating {num_events} mock DARPA events...")
        print(f"Output: {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate events with incrementing timestamps
        current_time = self.start_time
        
        with open(output_path, 'w') as f:
            for i in range(num_events):
                # Generate timestamp (advance 1-10 seconds)
                current_time += timedelta(seconds=random.uniform(1, 10))
                timestamp = self._generate_timestamp(current_time)
                
                # Generate event
                event = self.generate_event(timestamp)
                
                # Write to file
                f.write(json.dumps(event) + '\n')
                
                # Progress
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i + 1}/{num_events} events...")
        
        print(f"[SUCCESS] Generated {num_events} events")
        print(f"   Saved to: {output_path}")
        
        # Statistics
        file_size = output_path.stat().st_size
        print(f"   File size: {file_size / 1024:.2f} KB")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate Mock DARPA OpTC Dataset")
    parser.add_argument("--num-events", type=int, default=1000,
                       help="Number of events to generate")
    parser.add_argument("--output", type=str, default="data/dummy_darpa.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("="*60)
    print("MOCK DARPA DATASET GENERATOR")
    print("="*60)
    print(f"Seed: {args.seed}")
    print(f"Events: {args.num_events}")
    print()
    
    # Generate dataset
    generator = MockDARPAGenerator(seed=args.seed)
    output_path = generator.generate_dataset(args.num_events, args.output)
    
    print()
    print("="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Expected distribution:")
    for scenario, prob in generator.scenarios.items():
        count = int(args.num_events * prob)
        print(f"  {scenario}: ~{count} events ({prob*100:.0f}%)")
    
    print()
    print("[SUCCESS] Mock DARPA dataset ready for testing!")
    print(f"   Use: python scripts/train.py --dataset darpa_stream --darpa-path {output_path}")


if __name__ == "__main__":
    main()
