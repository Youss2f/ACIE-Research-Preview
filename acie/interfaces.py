"""
ACIE Interface Protocols (Federalism Layer)

This module defines the abstract base classes that enforce strict interoperability
across all dataset implementations. Any dataset class intended for use with the
ACIE training pipeline must inherit from BaseACIEDataset and implement the
required methods.

Architectural Principle: Interface Segregation (SOLID)
- All datasets adhere to a common contract
- Enables polymorphic usage across the pipeline
- Facilitates unit testing and mocking
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch


class BaseACIEDataset(ABC):
    """
    Abstract base class defining the contract for all ACIE-compatible datasets.
    
    This interface enforces three mandatory components:
    1. __len__: Returns the dataset cardinality
    2. __getitem__: Returns a (features, label) tuple at a given index
    3. input_dim: Property exposing the feature dimensionality
    
    All dataset implementations (SyntheticDataset, CyberLogDataset, DARPAStreamDataset)
    must inherit from this class and implement these methods to ensure
    compatibility with the ACIETrainer and ACIEEvaluator pipelines.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: The cardinality of the dataset.
        
        Note:
            For streaming datasets (IterableDataset), this may return an
            approximate count or require a full file scan.
        """
        raise NotImplementedError("Subclasses must implement __len__")
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample from the dataset.
        
        Args:
            idx: Integer index of the sample to retrieve.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - features: Tensor of shape (input_dim,) containing the feature vector
                - label: Tensor containing the class label (scalar or one-hot)
        
        Raises:
            IndexError: If idx is out of bounds.
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    @property
    @abstractmethod
    def input_dim(self) -> int:
        """
        Return the dimensionality of the input feature vectors.
        
        Returns:
            int: The number of features per sample.
        
        Note:
            This property is essential for model initialization,
            as ACIE_Core requires input_dim to configure its layers.
        """
        raise NotImplementedError("Subclasses must implement input_dim property")


class BaseACIEIterableDataset(ABC):
    """
    Abstract base class for streaming/iterable ACIE datasets.
    
    This interface is designed for datasets that cannot fit entirely in memory
    and must be streamed from disk (e.g., large DARPA OpTC JSONL files).
    
    Differences from BaseACIEDataset:
    - __iter__ replaces __getitem__ as the primary data access method
    - __len__ may return an approximate count or require expensive computation
    """
    
    @abstractmethod
    def __iter__(self):
        """
        Return an iterator over the dataset samples.
        
        Yields:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - features: Tensor of shape (input_dim,) containing the feature vector
                - label: Tensor containing the class label
        """
        raise NotImplementedError("Subclasses must implement __iter__")
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: The cardinality of the dataset (may be approximate for streaming).
        """
        raise NotImplementedError("Subclasses must implement __len__")
    
    @property
    @abstractmethod
    def input_dim(self) -> int:
        """
        Return the dimensionality of the input feature vectors.
        
        Returns:
            int: The number of features per sample.
        """
        raise NotImplementedError("Subclasses must implement input_dim property")
