import pytest
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from acie import ACIE_Core, ACIETrainer
from acie.dataset import CyberEventDataset


class TestACIETrainer:
    """Unit tests for ACIETrainer"""
    
    @pytest.fixture
    def model(self):
        """Create test model"""
        return ACIE_Core(input_dim=100, causal_nodes=10, action_space=5)
    
    @pytest.fixture
    def trainer(self, model):
        """Create test trainer"""
        return ACIETrainer(
            model=model,
            learning_rate=1e-3,
            lambda_dag=0.1,
            lambda_robust=0.5,
            device="cpu"
        )
    
    @pytest.fixture
    def dataloader(self):
        """Create test dataloader"""
        dataset = CyberEventDataset(num_samples=32, input_dim=100, num_classes=5)
        return DataLoader(dataset, batch_size=8, shuffle=True)
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initialization"""
        assert trainer.lambda_dag == 0.1
        assert trainer.lambda_robust == 0.5
        assert len(trainer.history["train_loss"]) == 0
    
    def test_compute_loss(self, trainer):
        """Test loss computation"""
        action_probs = torch.randn(4, 5)
        adjacency = torch.randn(10, 10)
        targets = torch.randint(0, 5, (4,))
        dag_constraint = torch.tensor(0.5)
        
        outputs = (action_probs, adjacency)
        loss, metrics = trainer.compute_loss(outputs, targets, dag_constraint)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert "policy_loss" in metrics
        assert "dag_loss" in metrics
    
    def test_train_epoch(self, trainer, dataloader):
        """Test single training epoch"""
        avg_loss, metrics = trainer.train_epoch(dataloader, epoch=1)
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
        assert "policy_loss" in metrics
    
    def test_validate(self, trainer, dataloader):
        """Test validation"""
        avg_loss, accuracy = trainer.validate(dataloader)
        
        assert isinstance(avg_loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
    
    def test_save_load_checkpoint(self, trainer, tmp_path):
        """Test checkpoint saving and loading"""
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        
        # Save checkpoint
        trainer.save_checkpoint(str(checkpoint_path), epoch=1, loss=0.5)
        assert checkpoint_path.exists()
        
        # Load checkpoint
        epoch, loss = trainer.load_checkpoint(str(checkpoint_path))
        assert epoch == 1
        assert loss == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
