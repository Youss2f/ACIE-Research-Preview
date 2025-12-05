import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy

class ACIE_Core(nn.Module):
    """
    The Unified Kernel: Integrates Information Theory (Filter), 
    Causal Inference (Graph), Game Theory (Policy), and Semantic Defense.
    
    The Semantic Consistency Layer provides defense against mimicry attacks
    (e.g., PROVNINJA) by enforcing domain-knowledge constraints on the
    learned causal graph.
    """
    def __init__(self, input_dim, causal_nodes, action_space):
        super().__init__()
        self.input_dim = input_dim
        self.nodes = causal_nodes
        
        # 1. Information Theoretic Layer (Compressive Sensing)
        # Random projection matrix for dimensionality reduction protecting against
        # adversarial noise (based on Restricted Isometry Property)
        self.sensing_matrix = nn.Parameter(torch.randn(input_dim // 10, input_dim), requires_grad=False)
        
        # 2. Causal Discovery Layer (Differentiable DAG Learning)
        # Adjacency matrix for the causal graph (A_ij > 0 implies i causes j)
        self.adjacency = nn.Parameter(torch.zeros(causal_nodes, causal_nodes))
        
        # 3. Semantic Consistency Layer (Anti-Mimicry Defense)
        # Fixed mask representing "Forbidden Transitions" based on domain knowledge
        # Violations are penalized heavily during training
        self.semantic_violation_mask = nn.Parameter(
            self._init_semantic_mask(causal_nodes), 
            requires_grad=False
        )
        
        # 4. Game Theoretic Layer (Policy Network)
        # Maps causal states to defense actions
        self.policy_net = nn.Sequential(
            nn.Linear(causal_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
    
    def _init_semantic_mask(self, num_nodes):
        """
        Initialize the semantic violation mask based on domain knowledge.
        
        Encodes "Forbidden Transitions" that should never appear in a valid
        causal attack graph. These represent semantically impossible paths
        that mimicry attacks might try to exploit.
        
        Example violations:
        - User Application (Node 3) directly accessing Database (Node 4) 
          without proper authentication chain
        - Exfiltration (Node 9) occurring before Lateral Movement (Node 4)
        - Direct jumps that skip required intermediate stages
        
        Returns:
            Tensor: Binary mask where 1.0 indicates a forbidden transition.
        """
        mask = torch.zeros(num_nodes, num_nodes)
        
        # Define semantically forbidden transitions
        # These represent domain knowledge about attack impossibilities
        
        # Rule 1: Cannot exfiltrate (9) before establishing presence (0,1)
        # Initial Access must precede Exfiltration in any valid chain
        mask[9, 0] = 1.0  # Exfil -> Initial Access (reverse causality)
        mask[9, 1] = 1.0  # Exfil -> Execution (reverse causality)
        
        # Rule 2: Privilege Escalation (3) cannot directly cause Initial Access (0)
        mask[3, 0] = 1.0  # PrivEsc -> Initial Access (impossible)
        
        # Rule 3: User-level processes (2) cannot directly write to kernel (7)
        # without privilege escalation
        mask[2, 7] = 1.0  # Persistence -> C2 kernel hook (requires PrivEsc)
        
        # Rule 4: Collection (6) cannot precede Lateral Movement (4)
        # in most attack scenarios (you collect after reaching target)
        mask[6, 4] = 1.0  # Collection -> Lateral (reverse order)
        
        # Rule 5: Self-loops are semantically invalid (no node causes itself)
        for i in range(num_nodes):
            mask[i, i] = 1.0
        
        # Rule 6: Impact (8) cannot directly lead to Defense Evasion (5)
        # Impact is typically a terminal action
        mask[8, 5] = 1.0
        mask[8, 1] = 1.0  # Impact -> Execution (impact is terminal)
        mask[8, 3] = 1.0  # Impact -> PrivEsc (impact is terminal)
        
        # =================================================================
        # CVE-2025-55182: React/Next.js RCE Detection Rules
        # =================================================================
        # These rules detect the insecure deserialization attack pattern:
        # HTTP_POST (high entropy payload) -> Deserialization -> Shell Spawn
        #
        # Rule 7: Web server (Node 5) directly spawning shell (Node 6) 
        # without intermediate application logic is HIGHLY SUSPICIOUS
        # This is the core CVE-2025-55182 pattern
        if num_nodes > 6:
            mask[5, 6] = 10.0  # MAXIMUM ALERT: Next.js -> Shell (RCE indicator)
        
        # Rule 8: Network event (Node 1) directly causing shell execution (Node 6)
        # without going through proper application flow
        if num_nodes > 6:
            mask[1, 6] = 5.0   # HIGH ALERT: Network -> Shell (potential RCE)
        
        # Rule 9: Deserialization context (if Node 5 is web server)
        # cannot directly lead to privilege escalation without shell
        if num_nodes > 5:
            mask[5, 3] = 2.0   # MEDIUM ALERT: Web -> PrivEsc (bypass attempt)
        
        return mask
    
    def compute_semantic_loss(self):
        """
        Compute the semantic violation loss.
        
        This penalizes the model if it tries to learn causal links that
        violate domain knowledge constraints. High semantic loss indicates
        the model is learning impossible attack paths (potential mimicry).
        
        Returns:
            Tensor: Scalar loss value representing semantic violations.
        """
        # Element-wise product: only penalize where mask=1 AND adjacency>0
        # Using abs(adjacency) to penalize both positive and negative weights
        violations = torch.abs(self.adjacency) * self.semantic_violation_mask
        semantic_loss = torch.sum(violations)
        
        return semantic_loss

    def information_filter(self, raw_stream):
        """
        Layer 1: Calculate Shannon Entropy to detect distribution shifts
        and gate processing to save compute/latency.
        """
        # Handle batch dimension properly
        batch_size = raw_stream.shape[0]
        
        # Process each sample in batch
        filtered_batch = []
        low_entropy_count = 0
        
        for i in range(batch_size):
            sample = raw_stream[i]
            
            # Discretize stream for entropy calc
            hist = torch.histc(sample, bins=100, min=-5, max=5)
            prob = hist / (torch.sum(hist) + 1e-9)
            h_current = -torch.sum(prob * torch.log(prob + 1e-9))
            
            # More permissive threshold for synthetic data
            if h_current < 2.5:  # Lowered threshold for better sensitivity
                low_entropy_count += 1
                # For low entropy, return zeros (will be filtered)
                filtered_batch.append(torch.zeros(self.sensing_matrix.shape[0], device=raw_stream.device))
            else:
                # Compressive Sensing Projection
                compressed = torch.matmul(self.sensing_matrix, sample)
                filtered_batch.append(compressed)
        
        result = torch.stack(filtered_batch)
        
        # Return result even if some samples are low entropy
        # Only return "filtered" status if ALL samples are low entropy
        return result, low_entropy_count == batch_size

    def causal_forward(self, compressed_data):
        """
        Layer 2: Infer causal structure using continuous optimization (NOTEARS-style).
        Returns the weighted adjacency matrix representing the attack path.
        """
        # Enforce acyclicity constraint trace(e^A) - d = 0
        h_acyclic = torch.trace(torch.matrix_exp(self.adjacency * self.adjacency)) - self.nodes
        
        # Forward pass to predict next state based on current causal graph
        causal_state = torch.matmul(compressed_data, self.adjacency)
        return causal_state, h_acyclic

    def robust_nash_policy(self, causal_state):
        """
        Layer 3: Select action that maximizes utility in Worst-Case scenario
        (Robust Optimization).
        """
        # Standard policy output
        action_logits = self.policy_net(causal_state)
        
        # Only apply adversarial perturbation during training
        if self.training and causal_state.requires_grad:
            # Adversarial perturbation (Game Theory): 
            # Assume attacker minimizes our utility within epsilon-ball
            perturbation = torch.sign(torch.autograd.grad(
                action_logits.mean(), causal_state, create_graph=True)[0])
            
            # Robust policy selection
            robust_logits = self.policy_net(causal_state - 0.1 * perturbation)
            return torch.softmax(robust_logits, dim=-1)
        else:
            # During evaluation, just return standard policy
            return torch.softmax(action_logits, dim=-1)

    def forward(self, event_stream):
        """
        Full forward pass through all four layers.
        
        Args:
            event_stream: Tensor of shape (batch_size, input_dim)
            
        Returns:
            tuple: (action_probs, adjacency, semantic_loss) or "NO_THREAT" if all samples filtered
        """
        # 1. Information Filter
        filtered, all_filtered = self.information_filter(event_stream)
        
        # Check if all samples were low-entropy
        if all_filtered:
            return "NO_THREAT"
            
        # 2. Causal Structure Discovery
        state, constraint = self.causal_forward(filtered)
        
        # 3. Semantic Consistency Check
        semantic_loss = self.compute_semantic_loss()
        
        # 4. Game Theoretic Response
        action_probs = self.robust_nash_policy(state)
        
        return action_probs, self.adjacency, semantic_loss
