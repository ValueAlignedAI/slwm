"""Output and diagnostic head stubs for Sprint I0."""

from models.heads.audio import AudioDecoderHead
from models.heads.base import BaseOutputHead
from models.heads.latent_prediction import LatentPredictionHead
from models.heads.noop import NoOpHead
from models.heads.reconstruction import ReconstructionHead
from models.heads.text import TextDecoderHead
from models.heads.uncertainty import UncertaintyHead
from models.heads.visual import VisualDecoderHead

__all__ = [
    "AudioDecoderHead",
    "BaseOutputHead",
    "LatentPredictionHead",
    "NoOpHead",
    "ReconstructionHead",
    "TextDecoderHead",
    "UncertaintyHead",
    "VisualDecoderHead",
]
