"""Visual/video adapter stub."""

from models.adapters.base import BaseModalityAdapter



class VisualSignalAdapter(BaseModalityAdapter):
    """Map image/video edge samples to ``z: FloatTensor[B,T,D]`` shape metadata.

    I0 does not create patches, tubelets, or visual codec latents.
    """

    modality = "visual_video"
