"""
DARPA OpTC Dataset Streaming Loader
Handles large-scale JSON-Lines files without loading entire dataset into memory.
"""

import json
import torch
from torch.utils.data import IterableDataset
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib

from acie.interfaces import BaseACIEIterableDataset


class DARPAStreamDataset(IterableDataset, BaseACIEIterableDataset):
    """
    Streaming dataset for DARPA OpTC JSON-Lines format.
    Reads massive .jsonl files line-by-line using generators for memory efficiency.
    
    Inherits from BaseACIEIterableDataset to ensure interface compliance.
    
    DARPA "Five Directions" Schema:
    - timestamp: Event occurrence time
    - actor: Source entity (IP, PID, etc.)
    - action: Event type (FLOW_START, PROCESS_CREATE, etc.)
    - object: Target entity
    - properties: Additional metadata
    """
    
    def __init__(
        self,
        jsonl_path: str,
        input_dim: int = 100,
        num_classes: int = 5,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        super().__init__()
        
        self.jsonl_path = Path(jsonl_path)
        self._input_dim = input_dim
        self.num_classes = num_classes
        self.max_samples = max_samples
        self.seed = seed
        
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"DARPA dataset not found: {jsonl_path}")
        
        # Field dimensions (must sum to input_dim)
        self.field_dims = {
            'timestamp': 10,
            'source_ip': 20,
            'dest_ip': 20,
            'process_id': 20,
            'event_type': 30
        }
        
        assert sum(self.field_dims.values()) == self._input_dim, \
            f"Field dimensions must sum to input_dim ({self._input_dim})"
        
        # Action type mapping (DARPA event types)
        self.action_mapping = {
            'FLOW_START': 1,          # NETWORK_CONNECT
            'FLOW_END': 1,
            'PROCESS_CREATE': 0,      # PROCESS_CREATE
            'PROCESS_TERMINATE': 8,   # PROCESS_TERMINATE
            'FILE_READ': 2,           # FILE_ACCESS
            'FILE_WRITE': 2,
            'FILE_DELETE': 2,
            'REGISTRY_READ': 3,       # REGISTRY_MODIFY
            'REGISTRY_WRITE': 3,
            'USER_LOGIN': 4,          # USER_LOGIN
            'PRIVILEGE_CHANGE': 5,    # PRIVILEGE_CHANGE
            'DATA_TRANSFER': 6,       # DATA_TRANSFER
            'SERVICE_START': 7,       # SERVICE_START
            'OTHER': 9                # OTHER
        }
        
        # Attack pattern detection heuristics (for labeling)
        self.attack_patterns = {
            0: "NO_THREAT",
            1: "LATERAL_MOVEMENT",
            2: "DATA_EXFILTRATION",
            3: "PRIVILEGE_ESCALATION",
            4: "MALWARE_EXECUTION"
        }
    
    def _hash_string_to_embedding(self, s: str, dim: int) -> np.ndarray:
        """
        Hash string to fixed-size embedding using hashing trick.
        Production ML technique for handling unbounded categorical features.
        """
        if not s:
            return np.zeros(dim)
        
        # Use MD5 hash for deterministic mapping
        hash_obj = hashlib.md5(s.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Create embedding by distributing hash across dimensions
        embedding = np.zeros(dim)
        for i in range(dim):
            # Use different hash offsets for each dimension
            idx = (hash_int + i * 31) % dim
            val = ((hash_int >> i) % 256) / 255.0  # Normalize to [0, 1]
            embedding[idx] = val - 0.5  # Center around 0
        
        return embedding
    
    def _parse_timestamp(self, timestamp_ns: int) -> np.ndarray:
        """
        Parse DARPA timestamp (nanoseconds since epoch) to feature vector.
        """
        features = np.zeros(self.field_dims['timestamp'])
        
        # Convert nanoseconds to seconds
        timestamp_sec = timestamp_ns / 1e9
        
        # Extract temporal features
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp_sec)
        
        # Hour of day (normalized)
        features[0] = dt.hour / 24.0
        
        # Day of week (normalized)
        features[1] = dt.weekday() / 7.0
        
        # Is weekend
        features[2] = 1.0 if dt.weekday() >= 5 else 0.0
        
        # Is off-hours (10pm - 6am)
        features[3] = 1.0 if dt.hour >= 22 or dt.hour <= 6 else 0.0
        
        # Minute fraction
        features[4] = dt.minute / 60.0
        
        # Add some hash-based features for uniqueness
        features[5:] = np.sin(timestamp_sec * np.array([1e-6, 1e-7, 1e-8, 1e-9, 1e-10]))
        
        return features
    
    def _parse_ip_address(self, ip_str: Optional[str], dim: int) -> np.ndarray:
        """
        Parse IP address to embedding using hashing trick.
        Handles IPv4, IPv6, and None cases.
        """
        if not ip_str:
            return np.zeros(dim)
        
        embedding = self._hash_string_to_embedding(ip_str, dim)
        
        # Add structured features if IPv4
        if '.' in ip_str:
            try:
                octets = [int(x) for x in ip_str.split('.')]
                if len(octets) == 4:
                    # Internal IP heuristic (10.x.x.x, 192.168.x.x, 172.16-31.x.x)
                    if octets[0] == 10 or (octets[0] == 192 and octets[1] == 168) or \
                       (octets[0] == 172 and 16 <= octets[1] <= 31):
                        embedding[0] = 1.0  # Internal flag
                    else:
                        embedding[0] = 0.0  # External flag
                    
                    # Encode octets (normalized)
                    embedding[1:5] = np.array(octets) / 255.0
            except:
                pass
        
        return embedding
    
    def _parse_action(self, action: str) -> np.ndarray:
        """
        Parse DARPA action to event type one-hot + features.
        """
        features = np.zeros(self.field_dims['event_type'])
        
        # Map to base event type (one-hot)
        event_idx = self.action_mapping.get(action, 9)  # Default to OTHER
        features[event_idx] = 1.0
        
        # Add action-specific features using hash
        action_embedding = self._hash_string_to_embedding(action, 10)
        features[10:20] = action_embedding
        
        return features
    
    def _detect_attack_pattern(self, event: Dict[str, Any]) -> int:
        """
        Heuristic-based attack pattern detection from DARPA event.
        Returns label (0-4) for training.
        """
        action = event.get('action', 'OTHER')
        
        # Extract IPs
        actor = event.get('actor', {})
        obj = event.get('object', {})
        src_ip = actor.get('src_ip') or actor.get('ip')
        dest_ip = obj.get('dest_ip') or obj.get('ip')
        
        # Heuristic 1: Lateral Movement
        # Both internal IPs + network connection
        if action in ['FLOW_START', 'FLOW_END']:
            if src_ip and dest_ip:
                src_internal = self._is_internal_ip(src_ip)
                dest_internal = self._is_internal_ip(dest_ip)
                if src_internal and dest_internal:
                    return 1  # LATERAL_MOVEMENT
        
        # Heuristic 2: Data Exfiltration
        # Internal -> External + large data transfer
        if action == 'DATA_TRANSFER' or 'FLOW' in action:
            if src_ip and dest_ip:
                src_internal = self._is_internal_ip(src_ip)
                dest_internal = self._is_internal_ip(dest_ip)
                if src_internal and not dest_internal:
                    return 2  # DATA_EXFILTRATION
        
        # Heuristic 3: Privilege Escalation
        if action == 'PRIVILEGE_CHANGE':
            return 3  # PRIVILEGE_ESCALATION
        
        # Heuristic 4: Malware Execution
        # Process create with suspicious properties
        if action == 'PROCESS_CREATE':
            properties = event.get('properties', {})
            if properties.get('suspicious', False):
                return 4  # MALWARE_EXECUTION
        
        # Default: No threat
        return 0
    
    def _is_internal_ip(self, ip_str: str) -> bool:
        """Check if IP is internal/private"""
        if not ip_str or '.' not in ip_str:
            return False
        try:
            octets = [int(x) for x in ip_str.split('.')]
            if len(octets) != 4:
                return False
            # 10.x.x.x, 192.168.x.x, 172.16-31.x.x
            return octets[0] == 10 or \
                   (octets[0] == 192 and octets[1] == 168) or \
                   (octets[0] == 172 and 16 <= octets[1] <= 31)
        except:
            return False
    
    def _parse_darpa_event(self, event: Dict[str, Any]) -> np.ndarray:
        """
        Parse DARPA JSON event to 100-dim feature vector.
        """
        feature_vector = np.zeros(self.input_dim)
        offset = 0
        
        # 1. Timestamp (10 dims)
        timestamp_ns = event.get('timestamp', 0)
        timestamp_features = self._parse_timestamp(timestamp_ns)
        feature_vector[offset:offset+self.field_dims['timestamp']] = timestamp_features
        offset += self.field_dims['timestamp']
        
        # 2. Source IP (20 dims)
        actor = event.get('actor', {})
        src_ip = actor.get('src_ip') or actor.get('ip')
        source_features = self._parse_ip_address(src_ip, self.field_dims['source_ip'])
        feature_vector[offset:offset+self.field_dims['source_ip']] = source_features
        offset += self.field_dims['source_ip']
        
        # 3. Destination IP (20 dims)
        obj = event.get('object', {})
        dest_ip = obj.get('dest_ip') or obj.get('ip')
        dest_features = self._parse_ip_address(dest_ip, self.field_dims['dest_ip'])
        feature_vector[offset:offset+self.field_dims['dest_ip']] = dest_features
        offset += self.field_dims['dest_ip']
        
        # 4. Process ID (20 dims)
        properties = event.get('properties', {})
        pid = properties.get('pid') or actor.get('pid')
        pid_str = str(pid) if pid else ""
        process_features = self._hash_string_to_embedding(pid_str, self.field_dims['process_id'])
        feature_vector[offset:offset+self.field_dims['process_id']] = process_features
        offset += self.field_dims['process_id']
        
        # 5. Event Type (30 dims)
        action = event.get('action', 'OTHER')
        event_features = self._parse_action(action)
        feature_vector[offset:offset+self.field_dims['event_type']] = event_features
        
        return feature_vector
    
    def _stream_events(self):
        """
        Generator that streams events from JSONL file line-by-line.
        Memory-efficient for massive files.
        """
        count = 0
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if self.max_samples and count >= self.max_samples:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    event = json.loads(line)
                    yield event
                    count += 1
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
    
    def __iter__(self):
        """
        Iterator for IterableDataset.
        Returns (features, label) tuples.
        """
        for event in self._stream_events():
            # Parse event to feature vector
            features = self._parse_darpa_event(event)
            
            # Detect attack pattern (label)
            label = self._detect_attack_pattern(event)
            
            # Convert to tensors
            features_tensor = torch.FloatTensor(features)
            label_tensor = torch.LongTensor([label])
            
            yield features_tensor, label_tensor.squeeze()
    
    def __len__(self) -> int:
        """
        Approximate length (requires full file scan - expensive).
        For IterableDataset, this is optional.
        """
        if self.max_samples:
            return self.max_samples
        
        # Count lines (cached for efficiency)
        count = 0
        with open(self.jsonl_path, 'r') as f:
            for _ in f:
                count += 1
        return count


class DARPABatchDataset(DARPAStreamDataset):
    """
    Batch-loading version of DARPA dataset.
    Loads chunks of data into memory for faster training.
    Use when you have sufficient RAM.
    """
    
    def __init__(
        self,
        jsonl_path: str,
        input_dim: int = 100,
        num_classes: int = 5,
        chunk_size: int = 10000,
        seed: int = 42
    ):
        super().__init__(jsonl_path, input_dim, num_classes, seed=seed)
        self.chunk_size = chunk_size
        
    def load_chunk(self, start_idx: int, end_idx: int):
        """Load a specific chunk of data"""
        features_list = []
        labels_list = []
        
        for idx, event in enumerate(self._stream_events()):
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break
            
            features = self._parse_darpa_event(event)
            label = self._detect_attack_pattern(event)
            
            features_list.append(features)
            labels_list.append(label)
        
        return torch.FloatTensor(features_list), torch.LongTensor(labels_list)
