"""
Training Script for ACIE Model

This script implements the governance layer for training, utilizing:
- ACIEConfig for centralized configuration management
- ACIERecorder for judicial oversight (logging and persistence)
- ACIETrainer for executive training execution
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split
import argparse
from pathlib import Path

from acie import ACIE_Core, ACIETrainer
from acie.dataset import SyntheticDataset, CyberLogDataset
from acie.darpa import DARPAStreamDataset
from acie.config import ACIEConfig
from acie.recorder import ACIERecorder
from acie.utils import set_seed, get_device, count_parameters


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ACIE Model")
    
    # Configuration file (takes precedence if provided)
    parser.add_argument("--config", type=str, default=None,
                       help="Path to JSON configuration file")
    
    # Model parameters
    parser.add_argument("--input-dim", type=int, default=100, help="Input dimension")
    parser.add_argument("--causal-nodes", type=int, default=10, help="Number of causal nodes")
    parser.add_argument("--action-space", type=int, default=5, help="Action space size")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda-dag", type=float, default=0.1, help="DAG constraint weight")
    parser.add_argument("--lambda-robust", type=float, default=0.5, help="Robustness weight")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="synthetic", 
                       choices=["synthetic", "cyber", "darpa_stream"],
                       help="Dataset type")
    parser.add_argument("--darpa-path", type=str, default="data/dummy_darpa.jsonl",
                       help="Path to DARPA JSONL file")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to load from streaming dataset")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--train-split", type=float, default=0.7, help="Training split ratio")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    
    # System parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    
    # Paths
    parser.add_argument("--save-dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--experiment-name", type=str, default="acie_v1", help="Experiment name")
    
    return parser.parse_args()


def load_config(args) -> ACIEConfig:
    """
    Load configuration from file or command-line arguments.
    
    Args:
        args: Parsed command-line arguments.
    
    Returns:
        ACIEConfig: Initialized configuration object.
    """
    if args.config and Path(args.config).exists():
        # Load from JSON file
        config = ACIEConfig.from_json(args.config)
        # Override with any explicit command-line arguments
        if args.experiment_name != "acie_v1":
            config.experiment_name = args.experiment_name
    else:
        # Build config from command-line arguments
        config = ACIEConfig.from_args(args)
    
    return config


def create_dataset(config: ACIEConfig):
    """
    Create dataset based on configuration.
    
    Args:
        config: ACIEConfig object.
    
    Returns:
        Dataset instance.
    """
    if config.data.dataset_type == "synthetic":
        return SyntheticDataset(
            num_samples=config.data.num_samples,
            input_dim=config.model.input_dim,
            num_classes=config.model.action_space,
            causal_nodes=config.model.causal_nodes,
            seed=config.seed
        )
    elif config.data.dataset_type == "cyber":
        return CyberLogDataset(
            num_samples=config.data.num_samples,
            input_dim=config.model.input_dim,
            num_classes=config.model.action_space,
            seed=config.seed
        )
    elif config.data.dataset_type == "darpa_stream":
        return DARPAStreamDataset(
            jsonl_path=config.data.darpa_path,
            input_dim=config.model.input_dim,
            num_classes=config.model.action_space,
            max_samples=config.data.max_samples,
            seed=config.seed
        )
    else:
        raise ValueError(f"Unknown dataset type: {config.data.dataset_type}")


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Load configuration (Governance Layer)
    config = load_config(args)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Initialize Recorder (Judicial Branch)
    recorder = ACIERecorder(
        experiment_name=config.experiment_name,
        save_dir=config.paths.model_dir,
        log_dir=config.paths.log_dir
    )
    
    # Log configuration summary
    recorder.log_info("=" * 60)
    recorder.log_info("ACIE Training Pipeline")
    recorder.log_info("=" * 60)
    recorder.log_info(config.summary())
    
    # Save configuration
    recorder.save_config(config.to_dict())
    
    # Get device
    if config.device == "auto":
        device = get_device()
    else:
        device = torch.device(config.device)
    
    recorder.log_info(f"Device: {device}")
    
    # Create dataset
    recorder.log_info("=" * 60)
    recorder.log_info("Loading Dataset")
    recorder.log_info("=" * 60)
    
    dataset = create_dataset(config)
    
    # Handle streaming vs. standard datasets
    if config.data.dataset_type == "darpa_stream":
        recorder.log_info("Using streaming dataset (no train/val split)")
        train_loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            num_workers=0  # Must be 0 for IterableDataset
        )
        val_loader = None
    else:
        recorder.log_info(f"Total samples: {len(dataset)}")
        
        # Split dataset
        train_size = int(config.data.train_split * len(dataset))
        val_size = int(config.data.val_split * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, _ = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        recorder.log_info(f"Train samples: {len(train_dataset)}")
        recorder.log_info(f"Val samples: {len(val_dataset)}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )
    
    # Create model
    recorder.log_info("=" * 60)
    recorder.log_info("Creating Model")
    recorder.log_info("=" * 60)
    
    model = ACIE_Core(
        input_dim=config.model.input_dim,
        causal_nodes=config.model.causal_nodes,
        action_space=config.model.action_space
    )
    
    num_params = count_parameters(model)
    recorder.log_info(f"Model created with {num_params:,} trainable parameters")
    
    # Create trainer (Executive Branch)
    recorder.log_info("=" * 60)
    recorder.log_info("Starting Training")
    recorder.log_info("=" * 60)
    
    trainer = ACIETrainer(
        model=model,
        learning_rate=config.training.learning_rate,
        lambda_dag=config.training.lambda_dag,
        lambda_robust=config.training.lambda_robust,
        device=device,
        recorder=recorder
    )
    
    # Execute training
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        save_path=str(recorder.save_dir / "best_model.pth")
    )
    
    # Save final model
    recorder.log_info("=" * 60)
    recorder.log_info("Saving Final Model")
    recorder.log_info("=" * 60)
    
    recorder.save_final_model(model)
    
    # Finalize recording session
    summary = recorder.finalize()
    
    recorder.log_info("=" * 60)
    recorder.log_info("Training Pipeline Complete")
    recorder.log_info("=" * 60)


if __name__ == "__main__":
    main()
