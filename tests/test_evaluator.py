import pytest
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from acie import ACIE_Core, ACIEEvaluator
from acie.dataset import CyberEventDataset


class TestACIEEvaluator:
    """Unit tests for ACIEEvaluator"""
    
    @pytest.fixture
    def model(self):
        """Create test model"""
        return ACIE_Core(input_dim=100, causal_nodes=10, action_space=5)
    
    @pytest.fixture
    def evaluator(self, model):
        """Create test evaluator"""
        return ACIEEvaluator(model, device="cpu")
    
    @pytest.fixture
    def dataloader(self):
        """Create test dataloader"""
        dataset = CyberEventDataset(num_samples=32, input_dim=100, num_classes=5)
        return DataLoader(dataset, batch_size=8, shuffle=False)
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator.device == torch.device("cpu")
        assert evaluator.model.training is False
    
    def test_evaluate(self, evaluator, dataloader):
        """Test evaluation"""
        metrics, (preds, targets, probs) = evaluator.evaluate(dataloader)
        
        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert len(preds) > 0
    
    def test_adversarial_robustness_test(self, evaluator, dataloader):
        """Test adversarial robustness"""
        robustness_results = evaluator.adversarial_robustness_test(
            dataloader,
            epsilon_values=[0.0, 0.1]
        )
        
        assert len(robustness_results) == 2
        assert 0.0 in robustness_results
        assert 0.1 in robustness_results
    
    def test_analyze_information_filter(self, evaluator, dataloader):
        """Test information filter analysis"""
        analysis, entropy_values = evaluator.analyze_information_filter(dataloader)
        
        assert "mean_entropy" in analysis
        assert "filtered_ratio" in analysis
        assert len(entropy_values) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
