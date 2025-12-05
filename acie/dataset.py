import torch
from torch.utils.data import Dataset
import numpy as np
import networkx as nx

from acie.interfaces import BaseACIEDataset


class SyntheticDataset(Dataset, BaseACIEDataset):
    """
    Synthetic dataset that generates event streams based on random causal DAGs.
    This simulates realistic cybersecurity scenarios where events have causal relationships.
    
    Inherits from BaseACIEDataset to ensure interface compliance.
    """
    def __init__(
        self,
        num_samples: int = 1000,
        input_dim: int = 100,
        num_classes: int = 5,
        causal_nodes: int = 10,
        dag_density: float = 0.3,
        noise_level: float = 0.1,
        seed: int = 42
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self._input_dim = input_dim
        self.num_classes = num_classes
        self.causal_nodes = causal_nodes
        
        # Generate a random DAG for each class (attack pattern)
        self.class_dags = []
        for class_id in range(num_classes):
            dag = self._generate_random_dag(causal_nodes, dag_density, seed + class_id)
            self.class_dags.append(dag)
        
        # Generate synthetic event streams
        self.event_streams = []
        self.labels = []
        
        for _ in range(num_samples):
            # Random class (attack type)
            label = np.random.randint(0, num_classes)
            
            # Generate event stream based on the causal DAG for this class
            stream = self._generate_causal_stream(
                self.class_dags[label], 
                input_dim, 
                noise_level
            )
            
            self.event_streams.append(stream)
            self.labels.append(label)
        
        self.event_streams = torch.FloatTensor(np.array(self.event_streams))
        self.labels = torch.LongTensor(self.labels)
    
    def _generate_random_dag(self, num_nodes, density, seed):
        """
        Generate a random Directed Acyclic Graph (DAG).
        Returns adjacency matrix representing causal relationships.
        """
        np.random.seed(seed)
        
        # Create random DAG using networkx
        G = nx.gnp_random_graph(num_nodes, density, seed=seed, directed=True)
        
        # Convert to DAG by removing cycles
        dag = nx.DiGraph()
        dag.add_nodes_from(G.nodes())
        
        # Only add edges that don't create cycles
        for u, v in G.edges():
            if u < v:  # Simple way to ensure acyclicity
                dag.add_edge(u, v, weight=np.random.uniform(0.5, 2.0))
        
        # Get adjacency matrix
        adj_matrix = nx.to_numpy_array(dag)
        
        return adj_matrix
    
    def _generate_causal_stream(self, dag_matrix, input_dim, noise_level):
        """
        Generate an event stream based on causal DAG structure.
        The DAG represents how features causally influence each other.
        """
        # Start with random noise
        stream = np.random.randn(input_dim) * noise_level
        
        # Map causal nodes to input dimensions
        nodes_per_feature = max(1, self.causal_nodes // 10)
        
        # Apply causal structure: if node i causes node j, reflect that in the stream
        for i in range(min(self.causal_nodes, input_dim)):
            for j in range(min(self.causal_nodes, input_dim)):
                if dag_matrix[i, j] > 0:
                    # Causal influence: feature i affects feature j
                    influence = dag_matrix[i, j] * stream[i]
                    if j < input_dim:
                        stream[j] += influence
        
        # Add class-specific signal
        stream += np.random.randn(input_dim) * 0.5
        
        return stream
    
    def get_class_dag(self, class_id):
        """Return the causal DAG for a specific class"""
        return self.class_dags[class_id]
    
    @property
    def input_dim(self) -> int:
        """Return the dimensionality of input feature vectors."""
        return self._input_dim
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.event_streams[idx], self.labels[idx]


class CyberEventDataset(Dataset, BaseACIEDataset):
    """
    Simple dataset for cybersecurity event streams (legacy compatibility).
    For causal graph-based generation, use SyntheticDataset instead.
    
    Inherits from BaseACIEDataset to ensure interface compliance.
    """
    def __init__(
        self,
        num_samples: int = 1000,
        input_dim: int = 100,
        num_classes: int = 5,
        noise_level: float = 0.1,
        seed: int = 42
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self._input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate synthetic event streams
        self.event_streams = []
        self.labels = []
        
        for _ in range(num_samples):
            # Random class
            label = np.random.randint(0, num_classes)
            
            # Generate event stream with class-specific patterns
            base_stream = np.random.randn(input_dim) + label * 0.5
            noise = np.random.randn(input_dim) * noise_level
            stream = base_stream + noise
            
            self.event_streams.append(stream)
            self.labels.append(label)
        
        self.event_streams = torch.FloatTensor(np.array(self.event_streams))
        self.labels = torch.LongTensor(self.labels)
    
    @property
    def input_dim(self) -> int:
        """Return the dimensionality of input feature vectors."""
        return self._input_dim
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.event_streams[idx], self.labels[idx]


class CyberLogDataset(Dataset, BaseACIEDataset):
    """
    Realistic cyber log dataset that simulates DARPA "Five Directions" schema.
    Generates structured log features representing cybersecurity events with:
    - Timestamp, Source_IP, Dest_IP, Process_ID, Event_Type
    - Simulates attack patterns like Lateral Movement
    
    Inherits from BaseACIEDataset to ensure interface compliance.
    """
    def __init__(
        self,
        num_samples: int = 1000,
        input_dim: int = 100,
        num_classes: int = 5,
        seed: int = 42
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self._input_dim = input_dim
        self.num_classes = num_classes
        
        # Define attack patterns
        self.attack_patterns = {
            0: "NO_THREAT",           # Normal activity
            1: "LATERAL_MOVEMENT",     # Node A -> Node B (database)
            2: "DATA_EXFILTRATION",    # Node B -> External IP
            3: "PRIVILEGE_ESCALATION", # Low privilege -> High privilege
            4: "MALWARE_EXECUTION",    # Process spawn chain
            5: "REACT_RCE_EXPLOIT"     # CVE-2025-55182: React/Next.js RCE via insecure deserialization
        }
        
        # Define cyber log field dimensions
        self.field_dims = {
            'timestamp': 10,      # Temporal features
            'source_ip': 20,      # Source IP embedding
            'dest_ip': 20,        # Destination IP embedding
            'process_id': 20,     # Process ID embedding
            'event_type': 30      # Event type one-hot + features
        }
        
        assert sum(self.field_dims.values()) == self._input_dim, \
            f"Field dimensions must sum to input_dim ({self._input_dim})"
        
        # Generate cyber logs
        self.logs = []
        self.labels = []
        
        for _ in range(num_samples):
            label = np.random.randint(0, num_classes)
            log = self._generate_cyber_log(label)
            
            self.logs.append(log)
            self.labels.append(label)
        
        self.logs = torch.FloatTensor(np.array(self.logs))
        self.labels = torch.LongTensor(self.labels)
    
    def _generate_cyber_log(self, attack_type):
        """
        Generate a realistic cyber log entry with attack-specific patterns
        """
        log = np.zeros(self.input_dim)
        offset = 0
        
        # 1. Timestamp (10 dims) - normalized time features
        timestamp = np.random.rand(self.field_dims['timestamp'])
        
        # Attack-specific temporal patterns
        if attack_type == 1:  # LATERAL_MOVEMENT
            # Rapid succession of events
            timestamp[0] = 0.9  # High activity indicator
            timestamp[1] = np.random.uniform(0.7, 1.0)  # Close time intervals
        elif attack_type == 2:  # DATA_EXFILTRATION
            # Off-hours activity
            timestamp[2] = 0.95  # Late night indicator
        elif attack_type == 5:  # REACT_RCE_EXPLOIT (CVE-2025-55182)
            # Rapid HTTP POST followed by shell spawn
            timestamp[0] = 0.95  # Very rapid event sequence
            timestamp[3] = 0.9   # HTTP request timing
            timestamp[4] = 0.85  # Process spawn timing (immediate)
        
        log[offset:offset+self.field_dims['timestamp']] = timestamp
        offset += self.field_dims['timestamp']
        
        # 2. Source IP (20 dims) - IP address embedding
        source_ip = np.random.randn(self.field_dims['source_ip']) * 0.3
        
        if attack_type == 1:  # LATERAL_MOVEMENT
            # Compromised Node A (192.168.1.100)
            source_ip[0] = 1.0  # Internal IP flag
            source_ip[1:5] = [0.192, 0.168, 0.001, 0.100]  # IP encoding
            source_ip[10] = 0.9  # Compromised flag
        elif attack_type == 2:  # DATA_EXFILTRATION
            # Internal Node B (192.168.1.200)
            source_ip[0] = 1.0
            source_ip[1:5] = [0.192, 0.168, 0.001, 0.200]
        elif attack_type == 5:  # REACT_RCE_EXPLOIT (CVE-2025-55182)
            # External attacker IP (malicious)
            source_ip[0] = 0.0   # External IP flag
            source_ip[1:5] = [0.185, 0.220, 0.101, 0.035]  # External attacker IP
            source_ip[17] = 0.95  # Known malicious IP indicator
            source_ip[18] = 0.9   # High threat score
        
        log[offset:offset+self.field_dims['source_ip']] = source_ip
        offset += self.field_dims['source_ip']
        
        # 3. Destination IP (20 dims)
        dest_ip = np.random.randn(self.field_dims['dest_ip']) * 0.3
        
        if attack_type == 1:  # LATERAL_MOVEMENT
            # Target Node B (Database: 192.168.1.200)
            dest_ip[0] = 1.0  # Internal IP flag
            dest_ip[1:5] = [0.192, 0.168, 0.001, 0.200]
            dest_ip[15] = 1.0  # Database server flag
        elif attack_type == 2:  # DATA_EXFILTRATION
            # External IP (8.8.8.8)
            dest_ip[0] = 0.0  # External IP flag
            dest_ip[1:5] = [0.008, 0.008, 0.008, 0.008]
            dest_ip[16] = 0.95  # Suspicious external connection
        elif attack_type == 5:  # REACT_RCE_EXPLOIT (CVE-2025-55182)
            # Next.js server (internal, ports 3000 or 443)
            dest_ip[0] = 1.0   # Internal IP flag
            dest_ip[1:5] = [0.192, 0.168, 0.001, 0.050]  # Next.js server IP
            dest_ip[6] = 0.95  # Port 3000 indicator (Next.js default)
            dest_ip[7] = 0.85  # Port 443 indicator (HTTPS)
            dest_ip[14] = 0.9  # Web server flag
            dest_ip[19] = 0.95 # Node.js runtime flag
        
        log[offset:offset+self.field_dims['dest_ip']] = dest_ip
        offset += self.field_dims['dest_ip']
        
        # 4. Process ID (20 dims) - process behavior embedding
        process_id = np.random.randn(self.field_dims['process_id']) * 0.2
        
        if attack_type == 1:  # LATERAL_MOVEMENT
            # SMB/RDP process signatures
            process_id[0] = 0.85  # Network service process
            process_id[5] = 0.9   # SMB/RDP indicator
        elif attack_type == 3:  # PRIVILEGE_ESCALATION
            # System process
            process_id[1] = 0.95  # System-level process
            process_id[10] = 0.8  # Elevated privileges
        elif attack_type == 4:  # MALWARE_EXECUTION
            # Suspicious process chain
            process_id[2] = 0.9   # Child process indicator
            process_id[12] = 0.85 # Suspicious behavior
        elif attack_type == 5:  # REACT_RCE_EXPLOIT (CVE-2025-55182)
            # Node.js spawning shell (critical RCE indicator)
            process_id[3] = 0.98  # Node.js process flag
            process_id[4] = 0.95  # Shell spawn indicator (sh/cmd.exe)
            process_id[13] = 0.99 # Process chain: node -> shell (CRITICAL)
            process_id[14] = 0.9  # Deserialization context flag
            process_id[15] = 0.85 # React Flight protocol indicator
        
        log[offset:offset+self.field_dims['process_id']] = process_id
        offset += self.field_dims['process_id']
        
        # 5. Event Type (30 dims) - one-hot encoded event + features
        event_type = np.zeros(self.field_dims['event_type'])
        
        # One-hot encoding for base event types (first 10 dims)
        base_event_types = [
            'PROCESS_CREATE', 'NETWORK_CONNECT', 'FILE_ACCESS',
            'REGISTRY_MODIFY', 'USER_LOGIN', 'PRIVILEGE_CHANGE',
            'DATA_TRANSFER', 'SERVICE_START', 'PROCESS_TERMINATE', 'OTHER'
        ]
        
        if attack_type == 0:  # NO_THREAT
            event_idx = np.random.choice([0, 2, 7, 9])  # Normal events
        elif attack_type == 1:  # LATERAL_MOVEMENT
            event_idx = 1  # NETWORK_CONNECT
            event_type[10:15] = [0.9, 0.85, 0.8, 0.75, 0.7]  # Connection features
            event_type[20] = 0.95  # Lateral movement signature
        elif attack_type == 2:  # DATA_EXFILTRATION
            event_idx = 6  # DATA_TRANSFER
            event_type[15:20] = [0.95, 0.9, 0.85, 0.8, 0.75]  # Large data transfer
            event_type[21] = 0.9  # Exfiltration signature
        elif attack_type == 3:  # PRIVILEGE_ESCALATION
            event_idx = 5  # PRIVILEGE_CHANGE
            event_type[22] = 0.95  # Privilege escalation signature
        elif attack_type == 4:  # MALWARE_EXECUTION
            event_idx = 0  # PROCESS_CREATE
            event_type[23] = 0.9  # Malware signature
        elif attack_type == 5:  # REACT_RCE_EXPLOIT (CVE-2025-55182)
            event_idx = 1  # NETWORK_CONNECT (HTTP_POST)
            # React Flight Protocol exploit signatures
            event_type[10] = 0.98  # HTTP POST indicator
            event_type[11] = 0.95  # High payload entropy (serialized data)
            event_type[12] = 0.92  # Content-Type: application/octet-stream
            event_type[13] = 0.99  # Deserialization trigger detected
            event_type[24] = 0.99  # CVE-2025-55182 signature (CRITICAL)
            event_type[25] = 0.95  # Shell spawn following HTTP request
            event_type[26] = 0.90  # Reverse shell establishment attempt
        
        event_type[event_idx] = 1.0  # One-hot encoding
        
        # Add noise to remaining features
        event_type[24:] += np.random.randn(6) * 0.1
        
        log[offset:offset+self.field_dims['event_type']] = event_type
        
        # Add small random noise to entire log
        log += np.random.randn(self.input_dim) * 0.05
        
        return log
    
    def get_attack_pattern(self, label):
        """Return the attack pattern name for a label"""
        return self.attack_patterns[label]
    
    @property
    def input_dim(self) -> int:
        """Return the dimensionality of input feature vectors."""
        return self._input_dim
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.logs[idx], self.labels[idx]


class RealWorldDataset(Dataset, BaseACIEDataset):
    """
    Template for loading real-world cybersecurity datasets
    (e.g., KDD Cup 99, NSL-KDD, CICIDS2017, DARPA OpTC)
    
    Inherits from BaseACIEDataset to ensure interface compliance.
    """
    def __init__(self, data_path: str, input_dim: int = 100, transform=None):
        self.data_path = data_path
        self._input_dim = input_dim
        self.transform = transform
        
        # TODO: Load your actual dataset here
        # self.data = pd.read_csv(data_path)
        # self.features, self.labels = self._preprocess()
        
        raise NotImplementedError("Please implement data loading for your specific dataset")
    
    def _preprocess(self):
        """
        Preprocess raw cybersecurity data
        """
        # TODO: Implement preprocessing
        # - Normalize features
        # - Encode categorical variables
        # - Handle missing values
        pass
    
    @property
    def input_dim(self) -> int:
        """Return the dimensionality of input feature vectors."""
        return self._input_dim
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
