"""
ACIE Trainer (Executive Branch)

This module implements the ACIETrainer class, responsible solely for the
training loop execution. All monitoring and persistence operations are
delegated to the ACIERecorder (Judicial Branch).

Architectural Principle: Single Responsibility (SOLID)
- Trainer handles optimization mechanics only
- Recording and logging delegated to ACIERecorder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Union
from tqdm import tqdm

from acie.recorder import ACIERecorder


class ACIETrainer:
    """
    Training pipeline for ACIE_Core with multi-objective optimization:
    1. Minimize prediction loss
    2. Enforce DAG constraint (acyclicity)
    3. Maximize robust policy performance
    4. Enforce semantic consistency (anti-mimicry defense)
    
    This class implements the Executive branch of the constitutional architecture,
    focusing purely on training execution while delegating monitoring to ACIERecorder.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        lambda_dag: float = 0.1,
        lambda_robust: float = 0.5,
        lambda_semantic: float = 1.0,
        device: Union[str, torch.device] = "auto",
        recorder: Optional[ACIERecorder] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The ACIE_Core model to train.
            learning_rate: Learning rate for Adam optimizer.
            lambda_dag: Weight for DAG acyclicity constraint.
            lambda_robust: Weight for robustness regularization.
            lambda_semantic: Weight for semantic consistency constraint.
            device: Device for training (auto, cuda, cpu, or torch.device).
            recorder: Optional ACIERecorder for logging and persistence.
                      If None, a minimal internal recorder is created.
        """
        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.lambda_dag = lambda_dag
        self.lambda_robust = lambda_robust
        self.lambda_semantic = lambda_semantic
        self.learning_rate = learning_rate
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        
        # Recorder (Judicial branch delegation)
        self.recorder = recorder
        
        # Internal history for backward compatibility
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "dag_constraint": [],
            "semantic_loss": [],
            "policy_accuracy": []
        }
    
    def compute_loss(self, outputs, targets, dag_constraint):
        """
        Multi-objective loss function combining:
        - Policy prediction loss
        - DAG acyclicity constraint
        - Semantic consistency constraint (anti-mimicry)
        - Sparsity regularization
        """
        if outputs == "NO_THREAT":
            return torch.tensor(0.0, device=self.device)
        
        action_probs, adjacency, semantic_loss = outputs
        
        # 1. Policy Loss
        policy_loss = self.policy_loss_fn(action_probs, targets)
        
        # 2. DAG Constraint (penalize cycles)
        dag_loss = self.lambda_dag * torch.abs(dag_constraint)
        
        # 3. Semantic Consistency Loss (anti-mimicry defense)
        semantic_loss_weighted = self.lambda_semantic * semantic_loss
        
        # 4. Sparsity regularization on adjacency matrix
        sparsity_loss = 0.01 * torch.norm(adjacency, p=1)
        
        total_loss = policy_loss + dag_loss + semantic_loss_weighted + sparsity_loss
        
        return total_loss, {
            "policy_loss": policy_loss.item(),
            "dag_loss": dag_loss.item(),
            "semantic_loss": semantic_loss.item(),
            "sparsity_loss": sparsity_loss.item()
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {"policy_loss": [], "dag_loss": [], "semantic_loss": [], "sparsity_loss": []}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (event_streams, targets) in enumerate(pbar):
            event_streams = event_streams.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(event_streams)
            
            # Handle NO_THREAT case
            if outputs == "NO_THREAT":
                continue
            
            action_probs, adjacency, semantic_loss = outputs
            
            # Get DAG constraint directly from adjacency
            dag_constraint = torch.trace(torch.matrix_exp(adjacency * adjacency)) - self.model.nodes
            
            # Compute loss
            loss, metrics = self.compute_loss(outputs, targets, dag_constraint)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            for key, val in metrics.items():
                epoch_metrics[key].append(val)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dag": f"{metrics['dag_loss']:.4f}",
                "sem": f"{metrics['semantic_loss']:.4f}"
            })
        
        # Average metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate(self, val_loader: DataLoader):
        """Validate the model"""
        self.model.eval()
        val_losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for event_streams, targets in val_loader:
                event_streams = event_streams.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(event_streams)
                
                if outputs == "NO_THREAT":
                    continue
                
                action_probs, adjacency, semantic_loss = outputs
                
                # Get DAG constraint directly from adjacency
                dag_constraint = torch.trace(torch.matrix_exp(adjacency * adjacency)) - self.model.nodes
                
                loss, _ = self.compute_loss(outputs, targets, dag_constraint)
                val_losses.append(loss.item())
                
                # Accuracy
                predictions = torch.argmax(action_probs, dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        avg_loss = np.mean(val_losses) if val_losses else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Full training loop.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            epochs: Number of training epochs.
            save_path: Path to save best model checkpoint.
        
        Returns:
            Dict: Training history.
        """
        if self.recorder:
            self.recorder.log_info(f"Starting training for {epochs} epochs on {self.device}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)
            
            # Validation
            val_loss = None
            val_acc = None
            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["policy_accuracy"].append(val_acc)
                
                # Save best model
                if save_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.recorder:
                        self.recorder.save_checkpoint(
                            self.model, self.optimizer, epoch, val_loss
                        )
                        self.recorder.record_best_model(epoch, val_loss)
                    else:
                        self.save_checkpoint(save_path, epoch, val_loss)
            
            # Record epoch metrics (delegate to recorder)
            if self.recorder:
                self.recorder.record_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_metrics=train_metrics,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    learning_rate=self.learning_rate
                )
        
        if self.recorder:
            self.recorder.log_info("Training completed")
        
        return self.history
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file.
        
        Returns:
            Tuple: (epoch, loss) from the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        if self.recorder:
            self.recorder.log_info(f"Loaded checkpoint from {path}")
        
        return checkpoint['epoch'], checkpoint['loss']
