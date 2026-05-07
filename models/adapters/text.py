"""Text/code adapter stub."""

from models.adapters.base import BaseModalityAdapter


class TextSignalAdapter(BaseModalityAdapter):
    """Map text/code edge samples to ``z: FloatTensor[B,T,D]`` shape metadata.

    I0 does not tokenize text or code. BPE/byte decisions are documented for
    later sprints only.
    """

    modality = "text_code"
