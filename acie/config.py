"""
ACIE Configuration Module (Governance Layer)

This module defines the ACIEConfig dataclass, which serves as the single source
of truth for all hyperparameters and system configurations. By centralizing
configuration in a strict, typed dataclass, we achieve:

1. Type Safety: Incorrect types are caught at initialization
2. Immutability: Configuration cannot be accidentally modified during training
3. Serialization: Easy conversion to/from JSON for experiment tracking
4. Documentation: All parameters are self-documenting via docstrings

Architectural Principle: Dependency Inversion
- High-level modules (Trainer, Evaluator) depend on abstractions (Config)
- Configuration can be injected, enabling easy testing and experimentation
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import json


@dataclass(frozen=False)
class ModelConfig:
    """Configuration for ACIE_Core model architecture."""
    input_dim: int = 100
    causal_nodes: int = 10
    action_space: int = 5


@dataclass(frozen=False)
class TrainingConfig:
    """Configuration for training hyperparameters."""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    lambda_dag: float = 0.1
    lambda_robust: float = 0.5
    lambda_semantic: float = 1.0  # Semantic consistency weight (anti-mimicry)
    weight_decay: float = 0.0
    gradient_clip: Optional[float] = None


@dataclass(frozen=False)
class DataConfig:
    """Configuration for dataset and data loading."""
    dataset_type: str = "synthetic"  # synthetic, cyber, darpa_stream
    num_samples: int = 1000
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    pin_memory: bool = True
    darpa_path: Optional[str] = None
    max_samples: Optional[int] = None


@dataclass(frozen=False)
class InformationFilterConfig:
    """Configuration for the information theoretic filter layer."""
    entropy_threshold: float = 3.5
    compression_ratio: int = 10
    bins: int = 100


@dataclass(frozen=False)
class CausalInferenceConfig:
    """Configuration for the causal discovery layer."""
    dag_constraint_weight: float = 0.1
    sparsity_weight: float = 0.01


@dataclass(frozen=False)
class GameTheoryConfig:
    """Configuration for the game theoretic policy layer."""
    epsilon_perturbation: float = 0.1
    robustness_weight: float = 0.5


@dataclass(frozen=False)
class PathConfig:
    """Configuration for file paths and directories."""
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    results_dir: str = "results"


@dataclass(frozen=False)
class ACIEConfig:
    """
    Master configuration object for the ACIE system.
    
    This dataclass aggregates all sub-configurations into a single,
    coherent configuration object. It provides methods for:
    - Loading from JSON files
    - Saving to JSON files
    - Converting to/from dictionaries
    - Validating configuration values
    
    Usage:
        # Create default configuration
        config = ACIEConfig()
        
        # Load from file
        config = ACIEConfig.from_json("configs/experiment.json")
        
        # Modify and save
        config.training.epochs = 200
        config.to_json("configs/new_experiment.json")
    """
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    information_filter: InformationFilterConfig = field(default_factory=InformationFilterConfig)
    causal_inference: CausalInferenceConfig = field(default_factory=CausalInferenceConfig)
    game_theory: GameTheoryConfig = field(default_factory=GameTheoryConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Experiment metadata
    experiment_name: str = "acie_experiment"
    seed: int = 42
    device: str = "auto"  # auto, cuda, cpu
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid.
        """
        # Validate split ratios
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(
                f"Data splits must sum to 1.0, got {total_split:.4f} "
                f"(train={self.data.train_split}, val={self.data.val_split}, test={self.data.test_split})"
            )
        
        # Validate positive values
        if self.training.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.training.learning_rate}")
        
        if self.training.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.training.batch_size}")
        
        if self.training.epochs <= 0:
            raise ValueError(f"Epochs must be positive, got {self.training.epochs}")
        
        if self.model.input_dim <= 0:
            raise ValueError(f"Input dimension must be positive, got {self.model.input_dim}")
        
        # Validate dataset type
        valid_datasets = {"synthetic", "cyber", "darpa_stream"}
        if self.data.dataset_type not in valid_datasets:
            raise ValueError(
                f"Invalid dataset type '{self.data.dataset_type}'. "
                f"Must be one of: {valid_datasets}"
            )
        
        # Validate device
        valid_devices = {"auto", "cuda", "cpu"}
        if self.device not in valid_devices:
            raise ValueError(
                f"Invalid device '{self.device}'. Must be one of: {valid_devices}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a nested dictionary.
        
        Returns:
            Dict: Nested dictionary representation of the configuration.
        """
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "information_filter": asdict(self.information_filter),
            "causal_inference": asdict(self.causal_inference),
            "game_theory": asdict(self.game_theory),
            "paths": asdict(self.paths),
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "device": self.device
        }
    
    def to_json(self, path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            path: Path to the output JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ACIEConfig":
        """
        Create configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values.
        
        Returns:
            ACIEConfig: Initialized configuration object.
        """
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            information_filter=InformationFilterConfig(**config_dict.get("information_filter", {})),
            causal_inference=CausalInferenceConfig(**config_dict.get("causal_inference", {})),
            game_theory=GameTheoryConfig(**config_dict.get("game_theory", {})),
            paths=PathConfig(**config_dict.get("paths", {})),
            experiment_name=config_dict.get("experiment_name", "acie_experiment"),
            seed=config_dict.get("seed", 42),
            device=config_dict.get("device", "auto")
        )
    
    @classmethod
    def from_json(cls, path: str) -> "ACIEConfig":
        """
        Load configuration from a JSON file.
        
        Args:
            path: Path to the JSON configuration file.
        
        Returns:
            ACIEConfig: Initialized configuration object.
        
        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_args(cls, args) -> "ACIEConfig":
        """
        Create configuration from argparse namespace.
        
        Args:
            args: Argparse namespace with configuration arguments.
        
        Returns:
            ACIEConfig: Initialized configuration object.
        """
        config = cls()
        
        # Model parameters
        if hasattr(args, 'input_dim'):
            config.model.input_dim = args.input_dim
        if hasattr(args, 'causal_nodes'):
            config.model.causal_nodes = args.causal_nodes
        if hasattr(args, 'action_space'):
            config.model.action_space = args.action_space
        
        # Training parameters
        if hasattr(args, 'batch_size'):
            config.training.batch_size = args.batch_size
        if hasattr(args, 'epochs'):
            config.training.epochs = args.epochs
        if hasattr(args, 'lr'):
            config.training.learning_rate = args.lr
        if hasattr(args, 'lambda_dag'):
            config.training.lambda_dag = args.lambda_dag
        if hasattr(args, 'lambda_robust'):
            config.training.lambda_robust = args.lambda_robust
        if hasattr(args, 'lambda_semantic'):
            config.training.lambda_semantic = args.lambda_semantic
        
        # Data parameters
        if hasattr(args, 'dataset'):
            config.data.dataset_type = args.dataset
        if hasattr(args, 'num_samples'):
            config.data.num_samples = args.num_samples
        if hasattr(args, 'train_split'):
            config.data.train_split = args.train_split
        if hasattr(args, 'val_split'):
            config.data.val_split = args.val_split
        if hasattr(args, 'num_workers'):
            config.data.num_workers = args.num_workers
        if hasattr(args, 'darpa_path'):
            config.data.darpa_path = args.darpa_path
        if hasattr(args, 'max_samples'):
            config.data.max_samples = args.max_samples
        
        # System parameters
        if hasattr(args, 'seed'):
            config.seed = args.seed
        if hasattr(args, 'device'):
            config.device = args.device
        if hasattr(args, 'experiment_name'):
            config.experiment_name = args.experiment_name
        
        # Paths
        if hasattr(args, 'save_dir'):
            config.paths.model_dir = args.save_dir
        
        # Revalidate after modifications
        config._validate()
        
        return config
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the configuration.
        
        Returns:
            str: Formatted configuration summary.
        """
        lines = [
            "=" * 60,
            "ACIE Configuration Summary",
            "=" * 60,
            f"Experiment: {self.experiment_name}",
            f"Device: {self.device}",
            f"Seed: {self.seed}",
            "",
            "Model:",
            f"  Input Dimension: {self.model.input_dim}",
            f"  Causal Nodes: {self.model.causal_nodes}",
            f"  Action Space: {self.model.action_space}",
            "",
            "Training:",
            f"  Batch Size: {self.training.batch_size}",
            f"  Epochs: {self.training.epochs}",
            f"  Learning Rate: {self.training.learning_rate}",
            f"  Lambda DAG: {self.training.lambda_dag}",
            f"  Lambda Robust: {self.training.lambda_robust}",
            f"  Lambda Semantic: {self.training.lambda_semantic}",
            "",
            "Data:",
            f"  Dataset Type: {self.data.dataset_type}",
            f"  Num Samples: {self.data.num_samples}",
            f"  Train/Val/Test Split: {self.data.train_split}/{self.data.val_split}/{self.data.test_split}",
            "",
            "=" * 60
        ]
        return "\n".join(lines)
