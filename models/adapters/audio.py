"""Audio adapter stub."""

from models.adapters.base import BaseModalityAdapter



class AudioSignalAdapter(BaseModalityAdapter):
    """Map audio edge samples to ``z: FloatTensor[B,T,D]`` shape metadata.

    I0 does not compute log-mel features, codecs, or waveform frontends.
    """

    modality = "audio"
