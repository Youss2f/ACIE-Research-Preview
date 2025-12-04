"""
ACIE: Adversarial Causal Intelligence Engine

A unified framework integrating Information Theory, Causal Inference, and Game Theory
for robust cyber threat detection and response.

Constitutional Architecture:
- Interfaces: Protocol definitions enforcing interoperability (Federalism)
- Config: Centralized configuration management (Governance)
- Recorder: Monitoring and persistence (Judicial Branch)
- Trainer: Training execution (Executive Branch)
- Core: Model implementation (Legislative Branch)
"""

__version__ = "0.2.0"
__author__ = "Youssef"

# Core components
from .core import ACIE_Core
from .trainer import ACIETrainer
from .evaluator import ACIEEvaluator

# Dataset implementations
from .dataset import SyntheticDataset, CyberEventDataset, CyberLogDataset
from .darpa import DARPAStreamDataset

# Constitutional architecture components
from .interfaces import BaseACIEDataset, BaseACIEIterableDataset
from .config import ACIEConfig, ModelConfig, TrainingConfig, DataConfig
from .recorder import ACIERecorder

__all__ = [
    # Core
    "ACIE_Core",
    "ACIETrainer", 
    "ACIEEvaluator",
    # Datasets
    "SyntheticDataset",
    "CyberEventDataset",
    "CyberLogDataset",
    "DARPAStreamDataset",
    # Interfaces
    "BaseACIEDataset",
    "BaseACIEIterableDataset",
    # Configuration
    "ACIEConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    # Recorder
    "ACIERecorder"
]
