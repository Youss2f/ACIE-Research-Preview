"""
ACIE Recorder (Judicial Branch)

This module implements the ACIERecorder class, responsible for all monitoring,
logging, and artifact persistence operations. By separating these concerns from
the ACIETrainer (Executive Branch), we achieve:

1. Single Responsibility: Each class has one reason to change
2. Testability: Recorder can be mocked for unit testing the trainer
3. Flexibility: Different recording backends can be swapped without modifying training logic

Architectural Principle: Separation of Concerns
- Trainer focuses on optimization mechanics
- Recorder handles observation and persistence
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch


class ACIERecorder:
    """
    Centralized recording and persistence system for ACIE training artifacts.
    
    Responsibilities:
    - Track training and validation metrics per epoch
    - Persist model checkpoints
    - Save training history to disk
    - Manage logging to file and console
    - Generate training summaries
    
    This class implements the Judicial branch of the constitutional architecture,
    providing oversight and record-keeping for the training process.
    """
    
    def __init__(
        self,
        experiment_name: str,
        save_dir: str = "models",
        log_dir: str = "logs",
        log_level: int = logging.INFO
    ):
        """
        Initialize the recorder with experiment metadata.
        
        Args:
            experiment_name: Unique identifier for this training run.
            save_dir: Directory for model checkpoints and artifacts.
            log_dir: Directory for log files.
            log_level: Logging verbosity level.
        """
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir) / experiment_name
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize history tracking
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "dag_constraint": [],
            "policy_accuracy": [],
            "learning_rate": []
        }
        
        # Metadata
        self.metadata: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_epochs": 0,
            "best_val_loss": float('inf'),
            "best_epoch": 0
        }
        
        # Configure logging
        self._setup_logging(log_level)
        
        self.logger.info(f"Recorder initialized for experiment: {experiment_name}")
        self.logger.info(f"Artifacts will be saved to: {self.save_dir}")
    
    def _setup_logging(self, log_level: int) -> None:
        """Configure logging handlers for file and console output."""
        self.logger = logging.getLogger(f"ACIE.{self.experiment_name}")
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # File handler
            log_file = self.log_dir / f"{self.experiment_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_metrics: Dict[str, float],
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None
    ) -> None:
        """
        Record metrics for a completed training epoch.
        
        Args:
            epoch: Current epoch number (1-indexed).
            train_loss: Average training loss for the epoch.
            train_metrics: Dictionary of additional training metrics.
            val_loss: Validation loss (if validation was performed).
            val_accuracy: Validation accuracy (if computed).
            learning_rate: Current learning rate.
        """
        self.history["train_loss"].append(train_loss)
        
        if "dag_loss" in train_metrics:
            self.history["dag_constraint"].append(train_metrics["dag_loss"])
        
        if val_loss is not None:
            self.history["val_loss"].append(val_loss)
        
        if val_accuracy is not None:
            self.history["policy_accuracy"].append(val_accuracy)
        
        if learning_rate is not None:
            self.history["learning_rate"].append(learning_rate)
        
        self.metadata["total_epochs"] = epoch
        
        # Log epoch summary
        log_msg = f"Epoch {epoch} - Train Loss: {train_loss:.4f}"
        if "dag_loss" in train_metrics:
            log_msg += f" - DAG: {train_metrics['dag_loss']:.4f}"
        if val_loss is not None:
            log_msg += f" - Val Loss: {val_loss:.4f}"
        if val_accuracy is not None:
            log_msg += f" - Val Acc: {val_accuracy:.4f}"
        
        self.logger.info(log_msg)
    
    def record_best_model(self, epoch: int, val_loss: float) -> None:
        """
        Update metadata when a new best model is found.
        
        Args:
            epoch: Epoch at which the best model was found.
            val_loss: Validation loss of the best model.
        """
        self.metadata["best_val_loss"] = val_loss
        self.metadata["best_epoch"] = epoch
        self.logger.info(f"New best model at epoch {epoch} with val_loss: {val_loss:.4f}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        filename: str = "best_model.pth"
    ) -> Path:
        """
        Save a model checkpoint to disk.
        
        Args:
            model: The model to save.
            optimizer: The optimizer state to save.
            epoch: Current epoch number.
            loss: Loss value at checkpoint.
            filename: Name of the checkpoint file.
        
        Returns:
            Path: The path to the saved checkpoint.
        """
        checkpoint_path = self.save_dir / filename
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'history': self.history,
            'metadata': self.metadata
        }, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path
    
    def save_final_model(self, model: torch.nn.Module) -> Path:
        """
        Save the final model state after training completion.
        
        Args:
            model: The trained model.
        
        Returns:
            Path: The path to the saved model.
        """
        model_path = self.save_dir / "final_model.pth"
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Final model saved to {model_path}")
        return model_path
    
    def save_history(self) -> Path:
        """
        Save the training history to a JSON file.
        
        Returns:
            Path: The path to the saved history file.
        """
        history_path = self.save_dir / "training_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        self.logger.info(f"Training history saved to {history_path}")
        return history_path
    
    def save_config(self, config: Dict[str, Any]) -> Path:
        """
        Save the configuration used for this experiment.
        
        Args:
            config: Configuration dictionary.
        
        Returns:
            Path: The path to the saved config file.
        """
        config_path = self.save_dir / "config.json"
        
        # Convert non-serializable types
        serializable_config = {}
        for key, value in config.items():
            if hasattr(value, '__dict__'):
                serializable_config[key] = value.__dict__
            else:
                serializable_config[key] = value
        
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=4, default=str)
        
        self.logger.info(f"Configuration saved to {config_path}")
        return config_path
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the recording session and generate summary.
        
        Returns:
            Dict: Summary of the training session.
        """
        self.metadata["end_time"] = datetime.now().isoformat()
        
        # Save final history
        self.save_history()
        
        # Generate summary
        summary = {
            "experiment": self.experiment_name,
            "total_epochs": self.metadata["total_epochs"],
            "best_epoch": self.metadata["best_epoch"],
            "best_val_loss": self.metadata["best_val_loss"],
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None,
            "duration": f"{self.metadata['start_time']} to {self.metadata['end_time']}"
        }
        
        # Save summary
        summary_path = self.save_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        self.logger.info("=" * 50)
        self.logger.info("Training Complete")
        self.logger.info(f"Best model at epoch {summary['best_epoch']}")
        self.logger.info(f"Best validation loss: {summary['best_val_loss']:.4f}")
        self.logger.info("=" * 50)
        
        return summary
    
    def get_history(self) -> Dict[str, List[float]]:
        """
        Return the current training history.
        
        Returns:
            Dict: Dictionary containing all tracked metrics.
        """
        return self.history.copy()
    
    def log_info(self, message: str) -> None:
        """Log an informational message."""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
