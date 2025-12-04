import torch
import logging
from pathlib import Path
import json


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """
    Setup logging configuration
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'acie.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_config(config: dict, save_path: str):
    """
    Save configuration to JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(config_path: str) -> dict:
    """
    Load configuration from JSON file
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def count_parameters(model):
    """
    Count trainable parameters in model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get appropriate device for computation
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
