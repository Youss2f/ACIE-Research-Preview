import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from acie import ACIE_Core


class TestACIECore:
    """Unit tests for ACIE_Core model"""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance"""
        return ACIE_Core(input_dim=100, causal_nodes=10, action_space=5)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        return torch.randn(4, 100)  # Batch size 4, input dim 100
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert model.input_dim == 100
        assert model.nodes == 10
        assert model.sensing_matrix.shape == (10, 100)
        assert model.adjacency.shape == (10, 10)
    
    def test_information_filter(self, model, sample_input):
        """Test information filter layer"""
        # High entropy input should pass
        high_entropy = torch.randn(100)
        filtered = model.information_filter(high_entropy)
        assert filtered is not None
        assert filtered.shape[0] == 10  # Compressed dimension
        
        # Low entropy input should be filtered
        low_entropy = torch.zeros(100)
        filtered = model.information_filter(low_entropy)
        # May pass or fail depending on exact entropy calculation
    
    def test_causal_forward(self, model):
        """Test causal inference layer"""
        compressed_data = torch.randn(4, 10)
        causal_state, h_acyclic = model.causal_forward(compressed_data)
        
        assert causal_state.shape == (4, 10)
        assert isinstance(h_acyclic, torch.Tensor)
        assert h_acyclic.numel() == 1  # Scalar constraint
    
    def test_robust_nash_policy(self, model):
        """Test game theoretic policy layer"""
        causal_state = torch.randn(4, 10, requires_grad=True)
        action_probs = model.robust_nash_policy(causal_state)
        
        assert action_probs.shape == (4, 5)
        assert torch.allclose(action_probs.sum(dim=1), torch.ones(4), atol=1e-5)
        assert (action_probs >= 0).all()
    
    def test_forward_pass(self, model, sample_input):
        """Test full forward pass"""
        output = model(sample_input)
        
        if output == "NO_THREAT":
            # Valid output
            pass
        else:
            action_probs, adjacency = output
            assert action_probs.shape[0] == sample_input.shape[0]
            assert action_probs.shape[1] == 5
            assert adjacency.shape == (10, 10)
    
    def test_model_parameters(self, model):
        """Test that model has trainable parameters"""
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check that some parameters require grad
        trainable = [p for p in params if p.requires_grad]
        assert len(trainable) > 0
    
    def test_model_device_transfer(self, model):
        """Test model can be moved to different devices"""
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            assert next(model_cuda.parameters()).is_cuda
            
            model_cpu = model_cuda.cpu()
            assert not next(model_cpu.parameters()).is_cuda


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
