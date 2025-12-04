import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy

class ACIE_Core(nn.Module):
    """
    The Unified Kernel: Integrates Information Theory (Filter), 
    Causal Inference (Graph), and Game Theory (Policy).
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
        
        # 3. Game Theoretic Layer (Policy Network)
        # Maps causal states to defense actions
        self.policy_net = nn.Sequential(
            nn.Linear(causal_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )

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
        Full forward pass through all three layers.
        
        Args:
            event_stream: Tensor of shape (batch_size, input_dim)
            
        Returns:
            tuple: (action_probs, adjacency) or "NO_THREAT" if all samples filtered
        """
        # 1. Information Filter
        filtered, all_filtered = self.information_filter(event_stream)
        
        # Check if all samples were low-entropy
        if all_filtered:
            return "NO_THREAT"
            
        # 2. Causal Structure Discovery
        state, constraint = self.causal_forward(filtered)
        
        # 3. Game Theoretic Response
        action_probs = self.robust_nash_policy(state)
        
        return action_probs, self.adjacency
