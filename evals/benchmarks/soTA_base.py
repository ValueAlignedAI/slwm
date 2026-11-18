from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

class SoTABase(ABC):
    """Abstract base class for standardizing evaluation protocols for SoTA models."""

    def __init__(self, model_path: Path):
        self.model_path = model_path

    @abstractmethod
    def load_model(self) -> None:
        """Initialize the model."""
        pass

    @abstractmethod
    def evaluate(self, dataset_path: Path) -> Dict[str, Any]:
        """Run standard evaluation protocol."""
        pass
